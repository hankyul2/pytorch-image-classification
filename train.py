from args import get_args
from data import get_data
from model import get_model
from setup import setup


def main(args):
    setup(args)
    train_dataloader, valid_dataloader = get_data(args)
    model, ema_model, ddp_model = get_model(args)


if __name__ == '__main__':
    args = get_args()

    main(args)