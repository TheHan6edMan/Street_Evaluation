import argparse

parser = argparse.ArgumentParser(description="Street Evaluation")

parser.add_argument(
    "--backbone",
    default="resnet50",
    type=str,
    help="Backbone module of semantic segmentation model"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/mnt/data/imagenet",
    help="The path where the data is stored. default:/mnt/data/imagenet"
)
parser.add_argument(

)

args = parser.parse_args()
