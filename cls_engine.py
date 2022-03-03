import torch

from metrics import reduce_mean


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
            loss = reduce_mean(loss, args.world_wize)

        if args.gpu == 0:
            loss_metric += loss.detach().item()

    print(f'valid loss: {loss_metric}')


def train_one_epoch(model, ema_model, ddp_model, optimizer, scheduler, criterion, scaler, args):
    pass