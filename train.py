from args import get_args
from data import get_data
from model import get_model
from optim_sched import get_optimizer_and_scheduler
from setup import setup


def main(args):
    setup(args)
    train_dataloader, valid_dataloader = get_data(args)
    model, ema_model, ddp_model = get_model(args)
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)


if __name__ == '__main__':
    args = get_args()

    main(args)