import pytest
import torch
from adversarial_attack.mnist_model import CnnNet, FCNet, load_mnist_data

@pytest.fixture(scope="session")
def device():
    """Fixture to provide the compute device for tests"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def fc_model(device):
    """Fixture to provide a trained fully connected model for tests"""
    model = FCNet()
    model.load_state_dict(torch.load('models/base_fc.pth'))
    model = model.to(device)
    model.eval()
    return model

@pytest.fixture(scope="session")
def cnn_model(device):
    """Fixture to provide a trained CNN model for tests"""
    model = CnnNet()
    model.load_state_dict(torch.load('models/base_cnn.pth'))
    model = model.to(device)
    model.eval()
    return model

@pytest.fixture(scope="session")
def data_loaders():
    """Fixture to provide training and test data loaders"""
    return load_mnist_data()
