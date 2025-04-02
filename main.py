import os
from adversarial_attack.mnist_model import load_mnist_data, load_pretrained_model, train_mnist
from adversarial_attack.generate import generate_adversarial
from adversarial_attack.visualize import visualize_multiple_adversarial, visualize_single_adversarial
import wandb
import yaml
import torch
from types import SimpleNamespace

def inference(model_path: str):
    wandb.init(project="Adversarial Attack", name="inference", mode="disabled")
    
    model = load_pretrained_model(model_path)
    model.eval()
    
    image, target, adversarial_target = __sample_all_numbers(model)
    
    # Load and convert hyperparameters to SimpleNamespace for attribute access
    config_dict = yaml.safe_load(open("hyper-parameters/hyperparameter.yaml"))
    config = SimpleNamespace(**config_dict)
    
    # Generate adversarial example
    perturbed_image = generate_adversarial(
        model, image, target_class=None, config=config, task='unsupervised'
    )
    adversarial_pred = model(perturbed_image).max(1).indices
    
    # Save images to file
    # for idx, (orig_img, pert_img) in enumerate(zip(image, perturbed_image)):
    #     orig_img_path = f"result/original_{idx}_truth_{target[idx]}.png"
    #     pert_img_path = f"result/adversarial_{idx}_truth_{target[idx]}_pred_{adversarial_pred[idx]}.png"
    #     plt.imsave(orig_img_path, orig_img.squeeze().cpu().numpy(), cmap='gray')
    #     plt.imsave(pert_img_path, pert_img.squeeze().cpu().numpy(), cmap='gray')
    
    # Visualize results
    image, perturbed_image = image.to('cpu'), perturbed_image.to('cpu')
    target, adversarial_pred = target.to('cpu'), adversarial_pred.to('cpu')
    for i in range(10):
        visualize_single_adversarial(image[i], perturbed_image[i], target[i], adversarial_pred[i], show=False)
    print(f"Original prediction: {target}")
    print(f"Adversarial prediction: {adversarial_pred}")
    print(f"True label: {target}")
    print(f"Average distortion: {__distortion(perturbed_image, image)}")

    wandb.finish()

def train_mnist():
    if not os.path.exists('models/base_fc.pth'):
        train_mnist('fc', 'models/base_fc.pth')
    if not os.path.exists('models/base_cnn.pth'):
        train_mnist('cnn', 'models/base_cnn.pth')

def __sample_all_numbers(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample one image for each number from the test dataset
    Making sure model's prediction is correct.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the images, original targets, and adversarial targets.
    """
    _, test_loader = load_mnist_data(batch_size=1)
    sampled_images = []
    sampled_targets = []
    sampled_adversarial_targets = []
    
    for image, target in test_loader:
        if target in sampled_targets:
            continue
        original_pred = model(image).max(1)[1].item()
        if original_pred != target:
            continue
        
        adversarial_target = (target + 1) % 10
        
        sampled_images.append(image)
        sampled_targets.append(target)
        sampled_adversarial_targets.append(adversarial_target)

        if len(sampled_images) == 10:
            break
    assert len(sampled_images) == 10, f"Only sampled {len(sampled_images)} images"
    return torch.cat(sampled_images), torch.tensor(sampled_targets), torch.tensor(sampled_adversarial_targets)

def __distortion(x, x_prime):
    return torch.sqrt(torch.mean((x - x_prime) ** 2))

def main():
    train_mnist()
    inference('models/base_fc.pth')

if __name__ == "__main__":
    main()