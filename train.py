from pic.utils import setup, get_args_parser, resume_from_checkpoint, print_metadata
from pic.data import get_dataset, get_dataloader
from pic.model import get_model
from pic.criterion import get_scaler_criterion
from pic.optimizer import get_optimizer_and_scheduler
from pic.use_case import validate, train_one_epoch


def main(args):
    # 0. init ddp & logger
    setup(args)

    # 1. load dataset
    train_dataset, valid_dataset = get_dataset(args)
    train_dataloader, valid_dataloader = get_dataloader(train_dataset, valid_dataset, args)

    # 2. make model
    model, ema_model, ddp_model = get_model(args)

    # 3. load optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    # 4. load criterion
    criterion, valid_criterion, scaler = get_scaler_criterion(args)

    # 5. print metadata
    print_metadata(model, train_dataset, valid_dataset, args)

    # 6. control logic for checkpoint & validate
    if args.resume:
        resume_from_checkpoint(args.checkpoint_path, model, ema_model, optimizer, scaler, scheduler)

    if args.validate_only:
        # Todo: improve validate loop
        validate(valid_dataloader, model, valid_criterion, args)
        if args.ema:
            validate(valid_dataloader, ema_model, valid_criterion, args)
        return

    start_epoch = args.start_epoch if args.start_epoch else 0
    end_epoch = (start_epoch + args.end_epoch) if args.end_epoch else args.epoch

    if scheduler is not None and start_epoch:
        # Todo: sequential lr does not support step with epoch as positional variable
        scheduler.step(start_epoch)

    # 7. train
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_metric = train_one_epoch(epoch, train_dataloader, ddp_model if args.distributed else model, optimizer, criterion, args,
                                       ema_model, scheduler, scaler)
        eval_metric = validate(valid_dataloader, model, valid_criterion, args)
        if args.ema:
            eval_ema_metric = validate(valid_dataloader, ema_model, valid_criterion, args)

        # Todo: save checkpoint
        # Todo: add logger


if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()

    main(args)