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

def layer_loss_v2(    
    model,
    x0: torch.Tensor,
    t: int,
    t_next: int, 
    e: torch.Tensor,
    b: torch.Tensor, 
    keepdim=False
):
    at = (1-b).cumprod(dim=0)[t].view(-1, 1, 1, 1)
    at_next = (1-b).cumprod(dim=0)[t_next].view(-1, 1, 1, 1)
    x = at.sqrt() * x0 + (1.0 - at).sqrt() * e
    xt_m1 = at_next.sqrt() * x0 + (1 - at_next).sqrt() * e
    output = model(x, t, t_next)
    coeff = at_next.sqrt() * (1 - at).sqrt() / at.sqrt() - (1 - at_next).sqrt()
    if keepdim:
        return (xt_m1 - output).square().sum(dim=(1, 2, 3))
    else:
        return ((xt_m1 - output) / coeff).square().sum(dim=(1, 2, 3)).mean(dim=0)

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



