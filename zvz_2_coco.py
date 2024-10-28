import csv
import os
import xml.etree.ElementTree as ET
import fiftyone as fo
from PIL import Image

# Barcode classes mapping according to
# "New Benchmarks for Barcode Detection Using Both Synthetic and Real Data" paper
barcode_classes = {
    "QRCode": "QRCode",
    "Aztec": "Aztec",
    "DataMatrix": "DataMatrix",
    "MaxiCode": "MaxiCode",
    "PDF417": "PDF417",

    "Code128": "Non-postal-1D-Barcodes",
    "Patch": "Non-postal-1D-Barcodes",
    "Industrial25": "Non-postal-1D-Barcodes",
    "EAN8": "Non-postal-1D-Barcodes",
    "EAN13": "Non-postal-1D-Barcodes",
    "Interleaved25": "Non-postal-1D-Barcodes",
    "Standard2of5": "Non-postal-1D-Barcodes",
    "Code32": "Non-postal-1D-Barcodes",
    "UCC128": "Non-postal-1D-Barcodes",
    "FullASCIICode": "Non-postal-1D-Barcodes",
    "MATRIX25": "Non-postal-1D-Barcodes",
    "Code39": "Non-postal-1D-Barcodes",
    "IATA25": "Non-postal-1D-Barcodes",
    "UPCA": "Non-postal-1D-Barcodes",
    "UPCE": "Non-postal-1D-Barcodes",
    "CODABAR": "Non-postal-1D-Barcodes",
    "Code93": "Non-postal-1D-Barcodes",
    "2-Digit": "Non-postal-1D-Barcodes",

    "Postnet": "Postal-1D-Barcodes",
    "AustraliaPost": "Postal-1D-Barcodes",
    "Kix": "Postal-1D-Barcodes",
    "IntelligentMail": "Postal-1D-Barcodes",
    "RoyalMailCode": "Postal-1D-Barcodes",
    "JapanPost": "Postal-1D-Barcodes",
}


def parse_annotation_xml(xml_path, img_width, img_height):
    """
    zvz dataset markup sample parser.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # list of points [[(x1, y1), (x2, y2), ..., (xn, yn)]]
    polygons = []
    categories = []

    for barcode in root.findall(".//Barcode"):
        points = [
            (
                float(point.attrib["X"]) / img_width,
                float(point.attrib["Y"]) / img_height,
            )
            for point in barcode.findall(".//Point")
        ]

        polygons.append([points])

        barcode_type = barcode.attrib["Type"]

        assert barcode_type in barcode_classes

        categories.append(barcode_classes[barcode_type])

    return polygons, categories


def convert_to_coco(idx, csv_file, dataset_dir, coco_output_path):
    """
    Converts zvz dataset to COCO object detection format.
    """
    dataset = fo.Dataset(name=f"my_dataset_{idx}")

    with open(csv_file, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            img_id, img_path, ann_path = row[:3]

            full_img_path = os.path.join(dataset_dir, img_path)

            with Image.open(full_img_path) as img:
                img_width, img_height = img.size

            sample = fo.Sample(filepath=full_img_path)

            ann_full_path = os.path.join(dataset_dir, ann_path)
            polygons, categories = parse_annotation_xml(
                ann_full_path, img_width, img_height
            )

            polylines = []
            for polygon, category in zip(polygons, categories):
                polyline = fo.Polyline(
                    label=category,
                    points=polygon,
                    closed=True,
                )
                polylines.append(polyline)

            sample["ground_truth"] = fo.Polylines(polylines=polylines)
            dataset.add_sample(sample)

    dataset.export(
        export_dir=coco_output_path,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        include_polylines=True,
    )


wholeDataset = [
    (
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real\split\split_f9_t0,1,2,3,4_seed42\dataset_train.csv",
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real",
        r"C:\temp\coco\real[train]",
    ),
    (
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real\split\split_f9_t0,1,2,3,4_seed42\dataset_valid.csv",
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real",
        r"C:\temp\coco\real[valid]",
    ),
    (
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real\split\split_f9_t0,1,2,3,4_seed42\dataset_infer.csv",
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real",
        r"C:\temp\coco\real[infer]",
    ),
    (
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real-512\split\split_f9_t0,1,2,3,4_seed42\dataset_train.csv",
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real-512",
        r"C:\temp\coco\real-512[train]",
    ),
    (
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real-512\split\split_f9_t0,1,2,3,4_seed42\dataset_valid.csv",
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real-512",
        r"C:\temp\coco\real-512[valid]",
    ),
    (
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real-512\split\split_f9_t0,1,2,3,4_seed42\dataset_infer.csv",
        r"C:\temp\BarcodeDetection\dataset\ZVZ-real-512",
        r"C:\temp\coco\real-512[infer]",
    ),
    (
        r"C:\temp\BarcodeDetection\dataset\ZVZ-synth-512\split\full\dataset_train.csv",
        r"C:\temp\BarcodeDetection\dataset\ZVZ-synth-512",
        r"C:\temp\coco\ZVZ-synth-512[train]",
    ),
    (
        r"C:\temp\BarcodeDetection\dataset\ZVZ-synth-512\split\full\dataset_valid.csv",
        r"C:\temp\BarcodeDetection\dataset\ZVZ-synth-512",
        r"C:\temp\coco\ZVZ-synth-512[valid]",
    ),
]

for idx, (csv_file, dataset_dir, coco_output_path) in enumerate(wholeDataset):
    convert_to_coco(idx, csv_file, dataset_dir, coco_output_path)
