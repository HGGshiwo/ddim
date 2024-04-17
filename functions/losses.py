import torch
from torchvision.utils import save_image

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
    true_xs = [x0]
    true_x = x0
    for i, j in zip(model.seq[1:], model.seq[:-1]):
        at = (1-b).cumprod(dim=0)[i].view(-1, 1, 1, 1)
        at_1 = (1-b).cumprod(dim=0)[j].view(-1, 1, 1, 1)
        true_x = (at/at_1).sqrt() * true_x + (1 - at/at_1).sqrt() * torch.randn_like(true_x)
        true_xs.append(true_x)

    x = true_xs[-1]
    loss = 0
    for i, j, true_x, next_true_x in zip(reversed(model.seq[1:]), reversed(model.seq[:-1]), reversed(true_xs[1:]), reversed(true_xs[:-1])):      
        x = model[str(i)].sample(x, i, j) 
        save_image((x[:16]+1)/2, f"{i}.png")
        save_image((true_x[:16]+1)/2, f"{i}_true.png")
        loss += (x - next_true_x).square()
    exit()
    if keepdim:
        return loss.sum(dim=(1, 2, 3))
    else:
        return loss.sum(dim=(1, 2, 3)).mean(dim=0)



