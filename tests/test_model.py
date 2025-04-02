"""
Tests for the MNIST model

Note: This test requires the pre-trained models 'models/base_fc.pth' and 'models/base_cnn.pth'
to be present in the repository.
"""


import pytest
import torch
import matplotlib.pyplot as plt
from adversarial_attack.mnist_model import FCNet, CnnNet, load_mnist_data

EXPECTED_ACCURACY = 0.90

@pytest.fixture
def fc_model():
    """Fixture to provide a trained FC model for tests"""
    model = FCNet()
    model.load_state_dict(torch.load('models/base_fc.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model

@pytest.fixture
def cnn_model():
    """Fixture to provide a trained CNN model for tests"""
    model = CnnNet()
    model.load_state_dict(torch.load('models/base_cnn.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model

@pytest.fixture
def data_loaders():
    """Fixture to provide training and test data loaders"""
    return load_mnist_data()

@pytest.mark.parametrize("model_fixture", ["fc_model", "cnn_model"])
def test_model_accuracy_on_train_set(model_fixture, request, data_loaders):
    """Test model accuracy on training set"""
    model = request.getfixturevalue(model_fixture)
    train_loader, _ = data_loaders
    device = next(model.parameters()).device
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    print(f"\n{model_fixture} Training accuracy: {accuracy:.4f}")
    assert accuracy >= EXPECTED_ACCURACY, f"Training accuracy {accuracy} is too low"

@pytest.mark.parametrize("model_fixture", ["fc_model", "cnn_model"])
def test_model_accuracy_on_test_set(model_fixture, request, data_loaders):
    """Test model accuracy on test set"""
    model = request.getfixturevalue(model_fixture)
    _, test_loader = data_loaders
    device = next(model.parameters()).device
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    print(f"\n{model_fixture} Test accuracy: {accuracy:.4f}")
    assert accuracy >= EXPECTED_ACCURACY, f"Test accuracy {accuracy} is too low"

@pytest.mark.parametrize("model_fixture", ["fc_model", "cnn_model"])
def test_model_single_prediction(model_fixture, request, data_loaders):
    """Test model prediction on a single image"""
    model = request.getfixturevalue(model_fixture)
    train_loader, _ = data_loaders
    device = next(model.parameters()).device
    
    # Get a single batch
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    
    # Get model prediction
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
    
    # Verify prediction shape and type
    assert pred.shape == (data.shape[0], 1), f"Unexpected prediction shape: {pred.shape}"
    assert pred.dtype == torch.int64, f"Unexpected prediction dtype: {pred.dtype}"

@pytest.mark.parametrize("model_fixture", ["fc_model", "cnn_model"])
def test_model_output_properties(model_fixture, request, data_loaders):
    """Test properties of model outputs"""
    model = request.getfixturevalue(model_fixture)
    train_loader, _ = data_loaders
    device = next(model.parameters()).device
    
    # Get a single batch
    data, _ = next(iter(train_loader))
    data = data.to(device)
    
    # Get model output
    with torch.no_grad():
        output = model(data)
    
    # Test output shape
    assert output.shape[0] == data.shape[0], "Batch size mismatch"
    assert output.shape[1] == 10, "Wrong number of output classes"
    
    # Test output properties
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    
    # Test probability distribution properties
    probs = torch.exp(output)
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs.sum(dim=1)), rtol=1e-5), \
        "Probabilities do not sum to 1"
