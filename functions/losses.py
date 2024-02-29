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

def layer_loss_alpha(    
    model,
    x0: torch.Tensor,
    t_m1: int,
    t: int,
    e: torch.Tensor,
    b: torch.Tensor, 
    keepdim=False
):
    # 使用e来计算x(t-1), 再使用e2来计算x(t), 损失是|f(x(t)) - x(t-1)|
    # 这样做保证e2和e是无关的, 如果先算x(t)会导致两个噪声有关联
    # 小心t和t-1不一定是相邻的
    if t_m1 == 0:
        x_m1 = x0
    else:
        a = (1-b).cumprod(dim=0)[t_m1].view(-1, 1, 1, 1)
        x_m1 = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    e2 = torch.randn_like(e)
    a2 = (1-b[t_m1:]).cumprod(dim=0)[t-t_m1].view(1, 1, 1, 1)
    x = x_m1 * a2.sqrt() + e2 * (1.0 - a2).sqrt()
    output = model(x, t)
    if keepdim:
        return (x_m1 - output).square().sum(dim=(1, 2, 3))
    else:
        return (x_m1 - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

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



