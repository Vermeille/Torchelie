from typing import Tuple
import torch


def zero_gp(model,
            real: torch.Tensor,
            fake: torch.Tensor,
            amp_scaler=None) -> Tuple[torch.Tensor, float]:
    """
    0-GP from Improving Generalization And Stability Of Generative Adversarial
    Networks ( https://arxiv.org/abs/1902.03984 ).

    Args:
        model (function / nn.Module): the model
        real (torch.Tensor): real images
        fake (torch.Tensor): fake images
        amp_scaler (torch.cuda.amp.GradScaler): if specified, will be used
            for computing the loss in fp16. Otherwise, use model's and
            data's dtype.

    Returns:
        A tuple (loss, gradient norm)
    """
    t = torch.rand(real.shape[0], 1, 1, 1, device=real.device, dtype=real.dtype)
    x = t * real + (1 - t) * fake
    return gradient_penalty(model, x, 0., amp_scaler)


def wgan_gp(model,
            real: torch.Tensor,
            fake: torch.Tensor,
            amp_scaler=None) -> Tuple[torch.Tensor, float]:
    """
    1-GP from Improved Training of Wasserstein GANs (
    https://arxiv.org/abs/1704.00028 ).

    Args:
        model (function / nn.Module): the model
        real (torch.Tensor): real images
        fake (torch.Tensor): fake images
        amp_scaler (torch.cuda.amp.GradScaler): if specified, will be used
            for computing the loss in fp16. Otherwise, use model's and
            data's dtype.

    Returns:
        A tuple (loss, gradient norm)
    """
    t = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
    t = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
    x = t * real + (1 - t) * fake
    return gradient_penalty(model, x, 1., amp_scaler)


def R1(model,
       real: torch.Tensor,
       fake: torch.Tensor,
       amp_scaler=None) -> Tuple[torch.Tensor, float]:
    """
    R1 regularizer from Which Training Methods for GANs do actually Converge?
    ( https://arxiv.org/abs/1801.04406 ).

    Args:
        model (function / nn.Module): the model
        real (torch.Tensor): real images
        fake (torch.Tensor): unused. Here for interface consistency with other
            penalties.
        amp_scaler (torch.cuda.amp.GradScaler): if specified, will be used
            for computing the loss in fp16. Otherwise, use model's and
            data's dtype.

    Returns:
        A tuple (loss, gradient norm)
    """
    return gradient_penalty(model, real, 0., amp_scaler)


def gradient_penalty(model,
                     data: torch.Tensor,
                     objective_norm: float,
                     amp_scaler=None) -> Tuple[torch.Tensor, float]:
    """
    Gradient penalty, mainly for GANs. Of the form
    :code:`E[(||dmodel(data)/ddata|| - objective_goal)²]`

    Args:
        model (function / nn.Module): the model
        data (torch.Tensor): input on which to measure the norm of the
            gradient
        objective_norm (float): objective norm. 1 for WGAN GP for instance.
        amp_scaler (torch.cuda.amp.GradScaler): if specified, will be used
            for computing the loss in fp16. Otherwise, use model's and
            data's dtype.

    Returns:
        A tuple (loss, gradient norm)
    """
    data = data.detach()
    data.requires_grad_(True)

    fp16 = amp_scaler is not None and amp_scaler.is_enabled()
    with torch.cuda.amp.autocast(enabled=fp16):
        out = model(data).sum()

    scale = amp_scaler.scale if fp16 else (lambda x: x)
    g = torch.autograd.grad(outputs=scale(out),
                            inputs=data,
                            create_graph=True,
                            only_inputs=True)[0]

    g = (g / amp_scaler.get_scale()) if fp16 else g

    g_norm = g.pow(2).sum(dim=(1, 2, 3)).add_(1e-8).sqrt()
    return (g_norm - objective_norm).pow(2).mean(), g_norm.mean().item()
