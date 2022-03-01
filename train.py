from args import get_args
from checkpoint import resume_from_checkpoint
from cls_engine import validate, train_one_epoch
from data import get_data
from model import get_model
from optim_sched_crit_scale import get_optimizer_and_scheduler, get_scaler_criterion
from setup import setup


def main(args):
    setup(args)
    train_dataloader, valid_dataloader = get_data(args)
    model, ema_model, ddp_model = get_model(args)
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)
    criterion, valid_criterion, scaler = get_scaler_criterion(args)

    if args.resume:
        resume_from_checkpoint(args.checkpoint_path, model, ema_model, optimizer, scaler, scheduler)

    start_epoch = args.start_epoch if args.start_epoch else 0
    end_epoch = (start_epoch + args.end_epoch) if args.end_epoch else args.epoch

    if scheduler is not None and start_epoch:
        # Todo: sequential lr does not support step with epoch as positional variable
        scheduler.step(start_epoch)

    if args.validate_only:
        validate(model, valid_criterion, valid_dataloader, args)
        if args.ema:
            validate(ema_model, valid_criterion, valid_dataloader, args)

    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_dataloader.set_epoch(epoch)

        train_metric = train_one_epoch(model, ema_model, ddp_model, optimizer, scheduler, criterion, scaler, args)
        eval_metric = validate(model, valid_criterion, valid_dataloader, args)
        if args.ema:
            eval_ema_metric = validate(ema_model, valid_criterion, valid_dataloader, args)

        # Todo: save checkpoint


if __name__ == '__main__':
    args = get_args()

    main(args)