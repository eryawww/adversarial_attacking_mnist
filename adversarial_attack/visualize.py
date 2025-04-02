import wandb
import matplotlib.pyplot as plt
import math
import torch

def visualize_single_adversarial(original_image, perturbed_image, original_pred, adversarial_pred, show:bool = False):
    """Visualize original and adversarial images side by side"""
    # Create a figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax[0].imshow(original_image.squeeze(), cmap='gray')
    ax[0].set_title(f'Original Predicted: {original_pred}')
    ax[0].axis('off')
    
    # Plot adversarial image
    ax[1].imshow(perturbed_image.squeeze(), cmap='gray')
    ax[1].set_title(f'Adversarial Predicted: {adversarial_pred}')
    ax[1].axis('off')
    
    # Plot perturbation
    difference = perturbed_image - original_image
    ax[2].imshow(difference.squeeze(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax[2].set_title('Perturbation (magnified)')
    ax[2].axis('off')
    
    # Save figure
    plt.savefig(f"result/adversarial_{original_pred}_pred_{adversarial_pred}.png")
    
    if show:
        plt.show()
    
    # Log the figure to wandb
    wandb.log({"Adversarial images": wandb.Image(fig)})
    
    # Log images to wandb
    wandb.log({
        "original_image": wandb.Image(original_image.squeeze()),
        "adversarial_image": wandb.Image(perturbed_image.squeeze()),
        "perturbation": wandb.Image(difference.squeeze(), caption="Perturbation (magnified)")
    })

def visualize_multiple_adversarial(original_images: torch.Tensor, perturbed_images: torch.Tensor, 
                                 original_preds: torch.Tensor, adversarial_preds: torch.Tensor, show:bool = False):
    """
    Visualize multiple pairs of original and adversarial images side by side
    
    Args:
        original_images: Tensor of shape (N, 1, 28, 28) containing original images
        perturbed_images: Tensor of shape (N, 1, 28, 28) containing perturbed images
        original_preds: Tensor of shape (N,) containing original predictions
        adversarial_preds: Tensor of shape (N,) containing adversarial predictions
    """
    batch_size = original_images.shape[0]
    
    # Calculate grid dimensions
    grid_size = batch_size
    fig = plt.figure(figsize=(7, 15))
    
    # For each image in the batch
    for idx in range(batch_size):
        # Get current image and predictions
        orig_img = original_images[idx]
        pert_img = perturbed_images[idx]
        orig_pred = original_preds[idx].item() if isinstance(original_preds, torch.Tensor) else original_preds[idx]
        adv_pred = adversarial_preds[idx].item() if isinstance(adversarial_preds, torch.Tensor) else adversarial_preds[idx]
        
        # Plot original image
        fig.add_subplot(grid_size, 3, idx * 3 + 1)
        plt.imshow(orig_img.squeeze(), cmap='gray')
        plt.xlabel(f'Original pred: {orig_pred}', fontsize=6)
        plt.xticks([])
        plt.yticks([])
        
        # Plot adversarial image
        fig.add_subplot(grid_size, 3, idx * 3 + 2)
        plt.imshow(pert_img.squeeze(), cmap='gray')
        plt.xlabel(f'Adversarial pred: {adv_pred}', fontsize=6)
        plt.xticks([])
        plt.yticks([])
        
        # Plot perturbation
        difference = pert_img - orig_img
        fig.add_subplot(grid_size, 3, idx * 3 + 3)
        plt.imshow(difference.squeeze(), cmap='RdBu', vmin=-0.5, vmax=0.5)
        plt.xticks([])
        plt.yticks([])
    
    plt.axis('off')
    plt.tight_layout()

    if show:
        plt.show()
    
    # Log the figure to wandb
    wandb.log({"Batch adversarial images": wandb.Image(fig)})
    
    # Log individual images to wandb
    for idx in range(batch_size):
        orig_img = original_images[idx]
        pert_img = perturbed_images[idx]
        difference = pert_img - orig_img
        
        wandb.log({
            f"batch/original_image_{idx}": wandb.Image(orig_img.squeeze()),
            f"batch/adversarial_image_{idx}": wandb.Image(pert_img.squeeze()),
            f"batch/perturbation_{idx}": wandb.Image(difference.squeeze(), caption=f"Perturbation {idx} (magnified)")
        })