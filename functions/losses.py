import torch


def layer_loss(
    model,
    x0: torch.Tensor,
    t: int,
    e: torch.Tensor,
    b: torch.Tensor, 
    keepdim=False
):
    a = (1-b).cumprod(dim=0)[t].view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def end2end_loss(
    model,
    x0: torch.Tensor,
    t: int,
    e: torch.Tensor,
    b: torch.Tensor, 
    keepdim=False
):
    a = (1-b).cumprod(dim=0)[t].view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model.sample(x)
    if keepdim:
        return (x0 - output).square().sum(dim=(1, 2, 3))
    else:
        return (x0 - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



