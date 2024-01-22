import torch.nn as nn


class LayerEMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadows = []

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        
        for block in module.models:
            shadow = {}
            for name, param in block.named_parameters():
                if param.requires_grad:
                    shadow[name] = param.data.clone()
            self.shadows.append(shadow)

    def update(self, module, t):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module[t].named_parameters():
            if param.requires_grad:
                self.shadows[t // module.skip][name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadows[t // module.skip][name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for i, block in enumerate(module.models):
            for name, param in block.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadows[i][name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadows

    def load_state_dict(self, state_dict):
        self.shadows = state_dict
