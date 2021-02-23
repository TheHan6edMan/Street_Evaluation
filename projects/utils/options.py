import argparse

parser = argparse.ArgumentParser(description="Street Evaluation")

parser.add_argument(
    "--backbone",
    default="resnet50",
    type=str,
    help="Backbone module of semantic segmentation model, default: resnet50"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/mnt/data/imagenet",
    help="The path where the data is stored, default:/mnt/data/imagenet"
)
parser.add_argument(
    "--arch",
    default="deeplabv3_resnet50",
    type=str,
    help="The architecture of the model for segmentation, default: deeplabv3_resnet50"
)

args = parser.parse_args()
