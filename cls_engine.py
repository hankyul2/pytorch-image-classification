import time

import torch

from metrics import Metric, Accuracy


@torch.inference_mode()
def validate(model, criterion, valid_dataloader, args):
    model.eval()
    data_m = Metric(reduce_every_n_step=args.print_freq, header='data')
    batch_m = Metric(reduce_every_n_step=args.print_freq, header='batch')
    top1_m = Metric(reduce_every_n_step=args.print_freq, header='top-1 acc')
    top5_m = Metric(reduce_every_n_step=args.print_freq, header='top-5 acc')
    loss_m = Metric(reduce_every_n_step=args.print_freq, header='loss')

    if args.channels_last:
        model = model.to(memory_format=args.channels_last)

    total_iter = len(valid_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(valid_dataloader):
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=args.channels_last)
            y = y.to(memory_format=args.channels_last)

        batch_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        top1, top5 = Accuracy(y_hat, y, top_k=(1,5,))

        top1_m.update(top1)
        top5_m.update(top5)
        loss_m.update(loss)

        if batch_idx and args.print_freq and args.print_freq % batch_idx == 0:
            args.log(f"[{batch_idx}/{total_iter}] {batch_m} {data_m} {loss_m} {top1_m} {top5_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    top1 = top1_m.compute()
    top5 = top5_m.compute()
    loss = loss_m.compute()
    batch = batch_m.compute()
    data = data_m.compute()

    args.log(f"{batch_m} {data_m} {loss_m} {top1_m} {top5_m}")

    return top1, top5, loss, batch, data


def train_one_epoch(model, ema_model, ddp_model, optimizer, scheduler, criterion, scaler, args):
    pass