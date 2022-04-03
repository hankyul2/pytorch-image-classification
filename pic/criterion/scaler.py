import torch
from torch.cuda.amp import GradScaler


class NativeScalerWithGradAccum:
    def __init__(self):
        """NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, scheduler=None, grad_norm=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if grad_norm:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_param, grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)