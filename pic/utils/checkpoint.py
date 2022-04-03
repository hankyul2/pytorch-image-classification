import os
import torch


def load_state_dict_from_checkpoint(checkpoint, key_list):
    for key in key_list:
        if checkpoint.get(key, None) is not None:
            return checkpoint.get(key)

    return None


def resume_from_checkpoint(checkpoint_path, model=None, ema_model=None, optimizer=None, scaler=None, scheduler=None):
    """resume training from checkpoint

    :arg
        checkpoint_path(str): checkpoint path
        model(nn.Module): model
        ema_model(nn.Module): ema model
        optimizer: optimizer
        scaler: pytorch native amp scaler
        scheduler: scheduler
    :return
        last epoch
    """
    obj_key_list = [(model, ['state_dict', 'model']), (ema_model, ['state_dict_ema', 'model_ema', 'state_dict', 'model']),
                    (optimizer, ['optimizer']), (scaler, ['scaler']), (scheduler, 'scheduler')]
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        for obj, key_list in filter(lambda x: x[0] is not None, obj_key_list):
            state_dict = load_state_dict_from_checkpoint(checkpoint, key_list)

            if state_dict:
                obj.load_state_dict(state_dict)
            elif 'state_dict' in key_list:
                obj.load_state_dict(checkpoint)
            else:
                raise ValueError(f'we can not find {key_list} in given checkpoint(dir={checkpoint_path})')

        return checkpoint.get('epoch', None)

    else:
        raise ValueError(f'no file exist in given checkpoint_path argument(dir={checkpoint_path}')


def save_checkpoint(save_dir, model, ema_model, optimizer, scaler, scheduler, epoch, is_best=False):
    pairs = [('state_dict',model), ('state_dict_ema', ema_model),
            ('optimizer', optimizer), ('scaler', scaler), ('scheduler', scheduler)]
    checkpoint_dict = {k:v.state_dict() for k, v in pairs if v}
    checkpoint_dict['epoch'] = epoch

    torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_{epoch}.pth'))
    torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_last.pth'))
    if is_best:
        torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_best.pth'))
    if os.path.exists(os.path.join(save_dir, f'checkpoint_{epoch-10}.pth')):
        os.remove(os.path.join(save_dir, f'checkpoint_{epoch-10}.pth'))





