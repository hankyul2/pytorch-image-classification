from pic.utils import setup, get_args_parser, save_checkpoint, resume_from_checkpoint, print_metadata
from pic.data import get_dataset, get_dataloader
from pic.model import get_model, get_ema_ddp_model
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
    model = get_model(args)
    model, ema_model, ddp_model = get_ema_ddp_model(model, args)

    # 3. load optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    # 4. load criterion
    criterion, valid_criterion, scaler = get_scaler_criterion(args)

    # 5. print metadata
    print_metadata(model, train_dataset, valid_dataset, args)

    # 6. control logic for checkpoint & validate
    if args.resume:
        start_epoch = resume_from_checkpoint(args.checkpoint_path, model, ema_model, optimizer, scaler, scheduler)
    else:
        start_epoch = 0

    start_epoch = args.start_epoch if args.start_epoch else start_epoch
    end_epoch = args.end_epoch if args.end_epoch else args.epoch

    if scheduler is not None and start_epoch:
        # Todo: sequential lr does not support step with epoch as positional variable
        scheduler.step(start_epoch)

    if args.validate_only:
        validate(valid_dataloader, model, valid_criterion, args, 'org')
        if args.ema:
            validate(valid_dataloader, ema_model, valid_criterion, args, 'ema')
        return

    # 7. train
    best_epoch = 0
    best_acc = 0

    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_dataloader, ddp_model if args.distributed else model, optimizer, criterion, args, ema_model, scheduler, scaler, epoch)
        val_loss, top1, top5 = validate(valid_dataloader, ddp_model if args.distributed else model, valid_criterion, args, 'org')
        if args.ema:
            eval_ema_metric = validate(valid_dataloader, ema_model.module, valid_criterion, args, 'ema')

        if args.use_wandb:
            args.log({'train_loss':train_loss, 'val_loss':val_loss, 'top1':top1, 'top5':top5}, metric=True)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        if args.is_rank_zero:
            save_checkpoint(args.log_dir, model, ema_model, optimizer,
                            scaler, scheduler, epoch, is_best=best_epoch == epoch)


if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()

    main(args)