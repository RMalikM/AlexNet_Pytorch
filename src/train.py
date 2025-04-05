import torch
import torch.nn as nn
from model import AlexNet
from dataset import get_train_valid_loader


def main(
        num_classes=10,
        num_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        model_save_path='best_model.pth'
    ):
    """
    Main function for training AlexNet on CIFAR-10 dataset and saving the best model 
    based on validation accuracy.

    This function trains the AlexNet model for a specified number of epochs, computes 
    validation accuracy after each epoch, and saves the model if the validation accuracy 
    improves. The model is saved as a `.pth` file at the specified location.

    Args:
        num_classes (int): Number of output classes for the model. Defaults to 10 (CIFAR-10).
        num_epochs (int): Number of epochs to train the model. Defaults to 50.
        batch_size (int): Number of samples per batch for DataLoader. Defaults to 64.
        learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
        model_save_path (str): Path where the best model based on validation accuracy will 
            be saved. Defaults to 'best_model.pth'.

    Workflow:
        - Initializes the model, dataset loaders, loss function, and optimizer.
        - Trains the model for the given number of epochs, computing loss at each step.
        - After each epoch, the model is evaluated on the validation set.
        - The model is saved to `model_save_path` if the validation accuracy improves.

    Notes:
        - The model is trained on the CIFAR-10 dataset.
        - The validation set is split from the training data, and validation accuracy 
          is computed after each epoch.
        - The model is saved as a `.pth` file if its performance improves during training.

    Example:
        main(num_classes=10, num_epochs=50, batch_size=64, learning_rate=0.001, 
             model_save_path='best_model.pth')

    """
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = AlexNet(num_classes).to(device)

    # CIFAR10 dataset 
    train_loader, valid_loader = get_train_valid_loader(
            data_dir = './data', 
            batch_size = batch_size,
            augment = False,
            random_seed = 1
        )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    best_valid_acc = 0.0  # To track the best validation accuracy

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

        # Validation after each epoch
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_acc = 100 * correct / total
        print(f'Accuracy of the network on the validation images: {valid_acc:.2f} %')

        # Save the model if validation accuracy improves
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved the best model with accuracy: {best_valid_acc:.2f} %')

    print(f'Best validation accuracy achieved: {best_valid_acc:.2f} %')


if __name__=='__main__':
    # Define the values of the parameters
    num_classes = 10
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

    main(
        num_classes,
        num_epochs,
        batch_size,
        learning_rate
    )