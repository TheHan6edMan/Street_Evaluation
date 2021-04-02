import argparse

__all__ = ["args"]

parser = argparse.ArgumentParser(description="Street Evaluation")

# data
parser.add_argument(
    "--data_path",
    type=str,
    default="/export/home/guest100/data",
    help="The path where the data is stored, default:None"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cityscapes",
    help="the dataset used"
)
parser.add_argument(
    "--ds_type",
    type=str,
    default="gtCoarse",
    help="the type of dataset to use, finely annotated or coarsely annotated, default: gtCoarse"
)
parser.add_argument(
    "--use_train_extra",
    type=bool,
    default=False,
    help="whether to use the `train_extra` set, default: False"
)
parser.add_argument(
    "--extract_zip",
    type=bool,
    default=False,
    help="whether to extract the zip file containing data, dft: False"
)


# models
parser.add_argument(
    "--backbone",
    default="resnet50",
    type=str,
    help="Backbone module of semantic segmentation model, default: resnet50"
)
parser.add_argument(
    "--arch",
    default="deeplabv3_resnet50",
    type=str,
    help="The architecture of the model for segmentation, default: deeplabv3_resnet50"
)
parser.add_argument(
    "--n_classes",
    type=int,
    default=19,
    help="number of classes, dft:19"
)


# training config
parser.add_argument(
    "--batch_size_train",
    type=int,
    default=8,
    help="batch size during training, dft: 8"
)
parser.add_argument(
    "--batch_size_test",
    type=int,
    default=16,
    help="batch size during training, dft: 16"
)
parser.add_argument(
    "--gpus",
    type=int,
    nargs='+',
    default=[0],
    help='the gpu_id to use. default:[0]',
)
parser.add_argument(
    "--n_workers",
    type=int,
    default=2,
    help="whether to use the `train_extra` set, default: False"
)
parser.add_argument(
    "--init_lr",
    type=float,
    default=0.01,
    help="the initial learning rate, dft:0.01"
)
parser.add_argument(
    "--ignore_index",
    type=int,
    default=-1,
    help="the `ignore_index` provided for `nn.CrossEntropyLoss`, dft:-1"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="the momentum for SGD algorithm, dft: 0.9"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="weight decay for SGD algorithm, dft:5e-4"
)
parser.add_argument(
    "--init_epoch",
    type=int,
    default=0,
    help="initial epoch, dft:0"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="the final epoch (excluded), default:200"
)
parser.add_argument(
    "--working_dir",
    type=str,
    default="../experiments/run"
)
parser.add_argument(
    "--baseline_dir",
    type=str,
    default="../experiments/baseline",

)

args = parser.parse_args()
