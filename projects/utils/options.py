import argparse

parser = argparse.ArgumentParser(description="Street Quality Evaluation")

parser.add_argument(
    "--pretrained_model_file",
    default=None,
    type=str,
    help="path to the checkpoint of the pretrained model"
)

