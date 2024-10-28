import os
import argparse
import cv2
import random

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetMapper,
)


# logging setup
setup_logger()

# The datasets are taken from https://github.com/abbyy/barcode_detection_benchmark
DATASETS = {
    "train": {
        "name": "real[train]",
        "json": "./real[train]/labels.json",
        "root": "./real[train]/data",
    },
    "valid": {
        "name": "real[valid]",
        "json": "./real[valid]/labels.json",
        "root": "./real[valid]/data",
    },
    "test": {
        "name": "real[infer]",
        "json": "./real[infer]/labels.json",
        "root": "./real[infer]/data",
    },
}

# There were a lot more classes in the initial datset.
# Some classes are merged according to "New Benchmarks for Barcode Detection Using Both Synthetic and Real Data" paper.
CLASSES = [
    "Aztec",
    "DataMatrix",
    "Non-postal-1D-Barcodes",
    "PDF417",
    "Postal-1D-Barcodes",
    "QRCode",
]


def register_datasets():
    """
    Appends datasets to the detectron2
    """
    for ds_type, ds_info in DATASETS.items():
        register_coco_instances(ds_info["name"], {}, ds_info["json"], ds_info["root"])
        MetadataCatalog.get(ds_info["name"]).set(thing_classes=CLASSES)


def configure_model(isEvaluation):
    """
    Creates configs for evaluation and training procesess.

    Returns:
        isEvaluation : is training or evaluation stage?
    """

    cfg = get_cfg()
    cfg.merge_from_file(
        "./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = [DATASETS["train"]["name"]]
    cfg.DATASETS.TEST = (
        [DATASETS["test"]["name"]] if isEvaluation else [DATASETS["valid"]["name"]]
    )
    cfg.DATALOADER.NUM_WORKERS = 8

    preTrainedModelPath = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.MODEL.WEIGHTS = (
        os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        if isEvaluation
        else preTrainedModelPath
    )

    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.BASE_LR_END = 0.0
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.MAX_ITER = 400
    cfg.SOLVER.WARMUP_ITERS = int(0.2 * cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.RESCALE_INTERVAL = True

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    cfg.MODEL.DEVICE = "cpu"

    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.INPUT.MIN_SIZE_TRAIN = 512
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.SOLVER.CHECKPOINT_PERIOD = 30

    evalPeriodInEpochs = 2
    cfg.TEST.EVAL_PERIOD = evalPeriodInEpochs * (
        len(DatasetCatalog.get(DATASETS["train"]["name"])) // cfg.SOLVER.IMS_PER_BATCH
    )

    cfg.MODEL.BACKBONE.FREEZE_AT = 4

    return cfg


class CustomTrainer(DefaultTrainer):
    """
    Model trainer and evaluator.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        """
        Creates evaluator for a COCO dataset.
        """
        return COCOEvaluator(dataset_name, cfg, False, output_dir)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Create data loader for training with custom augmentations.
        """
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=[
                    T.ResizeShortestEdge(
                        short_edge_length=[512], max_size=1024, sample_style="choice"
                    ),
                    T.RandomApply(T.RandomBrightness(0.7, 1.3), prob=0.05),
                    T.RandomApply(T.RandomContrast(0.7, 1.5), prob=0.05),
                    T.RandomApply(T.RandomSaturation(0.5, 1.5), prob=0.05),
                    T.RandomApply(T.RandomLighting(0.05), prob=0.05),
                    T.RandomApply(
                        T.RandomRotation(angle=[-30, 30], expand=True), prob=0.2
                    ),
                ],
            ),
        )

    @classmethod
    def build_test_loader(cls, cfg, name):
        """
        Create data loader for testing with custom augmentations.
        """
        return build_detection_test_loader(
            cfg,
            name,
            mapper=DatasetMapper(
                cfg,
                is_train=False,
                augmentations=[
                    T.ResizeShortestEdge(
                        short_edge_length=[512], max_size=1024, sample_style="choice"
                    ),
                ],
            ),
        )


def visualize_results(cfg):
    """
    Makes inference on some test samples and saves result images to temp directory.
    """

    dataset_dicts = DatasetCatalog.get(DATASETS["test"]["name"])
    metadata = MetadataCatalog.get(DATASETS["test"]["name"])

    predictor = DefaultPredictor(cfg)

    for idx, d in enumerate(random.sample(dataset_dicts, 50)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,
        )
        
        instances = outputs["instances"]
        high_conf_instances = instances[instances.scores > 0.5]
        
        v = v.draw_instance_predictions(high_conf_instances.to("cpu"))

        cv2.imwrite(rf"./temp/image_{idx}.jpg", v.get_image()[:, :, ::-1])


def main(skipTraining):
    """
    Trains a model (optional) and evaluates quality on the infer dataset.

    Args:
        skipTraining: skip the training process and run evaluation only.
    """
    register_datasets()

    if not skipTraining:
        trainCfg = configure_model(isEvaluation=False)
        trainer = CustomTrainer(trainCfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    evalCfg = configure_model(isEvaluation=True)
    visualize_results(evalCfg)
    evaluator = CustomTrainer(evalCfg)
    evaluator.resume_or_load(resume=False)
    evaluator.test(evalCfg, evaluator.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for training and testing a model."
    )
    parser.add_argument(
        "--skipTraining",
        action="store_true",
        help="Skips trainig process and evaluates current final model.",
    )
    args = parser.parse_args()
    main(args.skipTraining)
