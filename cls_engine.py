import torch
import torch.distributed as dist


@torch.inference_mode()
def validate(model, criterion, valid_dataloader, args):
    model.eval()
    loss_metric = 0

    for batch_idx, (x, y) in enumerate(valid_dataloader):
        x = x.to(args.device)
        y = y.to(args.device)

        y_hat = model(x)
        loss = criterion(y_hat, y)

        if args.distributed:
            dist.reduce(loss, 0, dist.ReduceOp.SUM)
            loss = loss / args.world_size

        if args.gpu == 0:
            loss_metric += loss.detach().item()

    print(f'valid loss: {loss_metric}')


def train_one_epoch(model, ema_model, ddp_model, optimizer, scheduler, criterion, scaler, args):
    pass