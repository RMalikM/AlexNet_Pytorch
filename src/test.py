import torch
from model import AlexNet
from dataset import get_test_loader


def load_model(model_path, num_classes=10):
    """
    Loads the saved model from the specified path.
    
    Args:
        model_path (str): Path to the saved model file.
        num_classes (int): Number of classes for the model. Defaults to 10 (CIFAR-10).
        
    Returns:
        torch.nn.Module: The loaded AlexNet model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def inference(model, test_loader):
    """
    Performs inference on the test dataset using the loaded model.

    Args:
        model (torch.nn.Module): The trained AlexNet model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        float: Test accuracy in percentage.
        list: A list of predicted labels for each test image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())  # Collect predictions

    accuracy = 100 * correct / total
    return accuracy, predictions

if __name__ == "__main__":
    # Load the saved model
    model_path = 'best_model.pth'  # Path to the saved model
    model = load_model(model_path, num_classes=10)

    # Get the test data loader
    test_loader = get_test_loader(data_dir='./data', batch_size=64)

    # Perform inference and calculate accuracy
    test_accuracy, test_predictions = inference(model, test_loader)

    # Output the results
    print(f'Accuracy of the model on the test images: {test_accuracy:.2f}%')
