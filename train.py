from args import get_args
from data import get_data
from setup import setup


def main(args):
    device = setup(args)
    train_dataloader, valid_dataloader = get_data(args)


if __name__ == '__main__':
    args = get_args()

    main(args)