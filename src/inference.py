import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from model import AlexNet  # Import your AlexNet model

# CIFAR-10 class labels
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

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

def process_image(image_path):
    """
    Process an image by loading it and applying the necessary transformations.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Transformed image tensor ready for inference.
    """
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # Resize image to the input size expected by AlexNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)  # Open the image using PIL
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension (since the model expects a batch of images)
    
    return image

def inference(model, image_dir):
    """
    Performs inference on each image in the specified directory and predicts classes.

    Args:
        model (torch.nn.Module): The trained AlexNet model.
        image_dir (str): Path to the directory containing images for inference.

    Returns:
        list: A list of tuples containing image file names and their predicted class labels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions = []

    # Iterate over all images in the directory
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        if os.path.isfile(image_path) and image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            # Process the image
            image = process_image(image_path).to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)  # Get predicted class index

            predicted_label = CIFAR10_CLASSES[predicted.item()]  # Map to class label
            predictions.append((image_file, predicted_label))  # Collect file name and prediction

    return predictions

if __name__ == "__main__":
    # Load the saved model
    model_path = 'best_model.pth'  # Path to the saved model
    model = load_model(model_path, num_classes=10)

    # Specify the directory containing the test images
    image_dir = './test_images'  # Directory with test images

    # Perform inference on images in the directory
    image_predictions = inference(model, image_dir)

    # Output the predicted labels for each image
    for image_file, pred_label in image_predictions:
        print(f'Image: {image_file}, Predicted class: {pred_label}')
