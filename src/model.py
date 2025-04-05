import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet model implementation for image classification.

    This class defines the architecture of the AlexNet model with modifications like 
    Batch Normalization, which helps in stabilizing the learning process. It consists of
    five convolutional layers followed by three fully connected layers.

    Attributes:
        layer1 (nn.Sequential): First convolutional layer with 96 filters, kernel size 11x11,
            stride 4, ReLU activation, and MaxPooling.
        layer2 (nn.Sequential): Second convolutional layer with 256 filters, kernel size 5x5,
            stride 1, padding 2, ReLU activation, and MaxPooling.
        layer3 (nn.Sequential): Third convolutional layer with 384 filters, kernel size 3x3,
            stride 1, padding 1, and ReLU activation.
        layer4 (nn.Sequential): Fourth convolutional layer with 384 filters, kernel size 3x3,
            stride 1, padding 1, and ReLU activation.
        layer5 (nn.Sequential): Fifth convolutional layer with 256 filters, kernel size 3x3,
            stride 1, padding 1, ReLU activation, and MaxPooling.
        fc1 (nn.Sequential): First fully connected layer with 4096 neurons, ReLU activation,
            and Dropout with a probability of 0.5.
        fc2 (nn.Sequential): Second fully connected layer with 4096 neurons, ReLU activation,
            and Dropout with a probability of 0.5.
        fc3 (nn.Sequential): Final fully connected layer mapping to the number of output 
            classes.

    Args:
        num_classes (int): Number of output classes for classification. Defaults to 1000.

    Forward Method:
        The `forward` method defines the forward pass through the network. The input image 
        passes through the convolutional layers, gets flattened, and is fed through the fully 
        connected layers to output the class predictions.

    Returns:
        torch.Tensor: The output tensor with shape (batch_size, num_classes), containing 
        the classification scores for each class.
    """

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc3= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out