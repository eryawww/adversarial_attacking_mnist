import torch
import torch.nn.functional as F
import wandb
from typing import Callable, Optional, Tuple, Literal

def generate_adversarial(model, image, target_class: Optional[torch.Tensor] = None, config=None, task: Literal['supervised', 'unsupervised']='unsupervised'):
    """Generate adversarial examples using either supervised or unsupervised method
    
    Args:
        model: Target model to attack
        image: Input image tensor (BATCH, 1, 28, 28)
        target_class: Target class for supervised mode, ignored in unsupervised mode
        config: Configuration with lr, c, and max_iterations
        task: 'supervised' or 'unsupervised'
    Returns:
        Perturbed image tensor
    """
    if task == 'supervised' and target_class is None:
        raise ValueError("target_class required for supervised mode")
        
    loss_fn = {
        'supervised': lambda model, img, r: __compute_supervised_loss(model, img, r, target_class),
        'unsupervised': __compute_unsupervised_loss
    }.get(task)
    
    if loss_fn is None:
        raise ValueError(f"Invalid task: {task}")
        
    return __generate_adversarial_base(model, image, loss_fn, config)

def __generate_adversarial_base(model, image, loss_fn: Callable, config) -> torch.Tensor:
    """Base adversarial example generation function
    
    Args:
        model: Target model
        image: Input image
        loss_fn: Loss function that takes (model, image, perturbation)
        config: Hyperparameters
    Returns:
        Perturbed image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    image = image.to(device)
    
    # Initialize perturbation
    r = torch.zeros_like(image, requires_grad=True)
    optimizer = torch.optim.Adam([r], lr=config.lr)
    
    for i in range(config.max_iterations):
        optimizer.zero_grad()
        
        perturbation_loss, model_loss = loss_fn(model, image, r)
        loss = config.c * perturbation_loss + model_loss
        
        loss.backward()
        optimizer.step()
            
        if i % 100 == 0:
            __log_metrics(r, perturbation_loss, model_loss, loss, i)
    
    perturbed_image = F.sigmoid(image + r)
    return perturbed_image.detach()

def __compute_supervised_loss(model, image, r, target_class) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute supervised adversarial loss
    
    Returns:
        Tuple of (perturbation_loss, model_loss)
    """
    perturbation_loss = torch.norm(r, p=2)
    perturbed_image = F.sigmoid(image + r)
    model_loss = F.nll_loss(model(perturbed_image), target_class)
    return perturbation_loss, model_loss

def __compute_unsupervised_loss(model, image, r) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute unsupervised adversarial loss
    
    Returns:
        Tuple of (perturbation_loss, model_loss)
    """
    perturbation_loss = torch.norm(r, p=2)
    perturbed_image = F.sigmoid(image + r)
    model_output = model(perturbed_image)
    # Maximize negative NLL loss, we want them to be wrong
    model_loss = -1 * F.nll_loss(model_output, torch.argmax(model_output, dim=1))
    return perturbation_loss, model_loss

def __log_metrics(r: torch.Tensor, perturbation_loss: torch.Tensor, 
                 model_loss: torch.Tensor, total_loss: torch.Tensor, iteration: int):
    """Log metrics to wandb"""
    with torch.no_grad():
        r_mean = torch.mean(r.clone().detach(), dim=0)
        r_max = torch.norm(r.clone().detach(), p=float('inf'))
        
        wandb.log({
            'perturbation_loss': perturbation_loss.item(),
            'model_loss': model_loss.item(),
            'total_loss': total_loss.item(),
            'r_norm_inf': r_max.item(),
            'grad_norm': torch.norm(r_mean, p=2).item() if r.grad is not None else 0,
            'iteration': iteration,
            'perturbation_norm': torch.norm(r_mean, p=2).item()
        })