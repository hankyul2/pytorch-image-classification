import datetime
import time

import torch

from pic.utils import Metric, Accuracy, reduce_mean


@torch.inference_mode()
def validate(valid_dataloader, model, criterion, args, mode='org'):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    top1_m = Metric(reduce_every_n_step=args.print_freq, header='Top-1:')
    top5_m = Metric(reduce_every_n_step=args.print_freq, header='Top-5:')
    loss_m = Metric(reduce_every_n_step=args.print_freq, header='Loss:')

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(valid_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        top1, top5 = Accuracy(y_hat, y, top_k=(1,5,))

        top1_m.update(top1, batch_size)
        top5_m.update(top5, batch_size)
        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"VALID({mode}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m} {top1_m} {top5_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    top1 = top1_m.compute()
    top5 = top5_m.compute()
    loss = loss_m.compute()

    # 4. print metric
    space = 16
    num_metric = 6
    args.log('-'*space*num_metric)
    args.log(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Top-1 Acc', 'Top-5 Acc'))
    args.log('-'*space*num_metric)
    args.log(f"{'VALID('+mode+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{top1:{space}.4f}{top5:{space}.4f}")
    args.log('-'*space*num_metric)

    return batch_m.sum, data_m.sum, top1, top5, loss



def train_one_epoch(epoch, train_dataloader, model, optimizer, criterion, args, ema_model=None, scheduler=None, scaler=None,):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    loss_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Loss:')

    # 2. start validate
    model.train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(train_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        if args.distributed:
            loss = reduce_mean(loss, args.world_size)

        if args.amp:
            scaler(loss, optimizer, model.parameters(), args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    loss = loss_m.compute()

    # 4. print metric
    space = 16
    num_metric = 4
    args.log('-'*space*num_metric)
    args.log(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss'))
    args.log('-'*space*num_metric)
    args.log(f"{'TRAIN('+epoch+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{loss:>{space}}")
    args.log('-'*space*num_metric)

    return batch_m.sum, data_m.sum, loss
