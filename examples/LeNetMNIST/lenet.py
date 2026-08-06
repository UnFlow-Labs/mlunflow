# Load in relevant libraries, and alias where appropriate
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pydantic import BaseModel, ConfigDict, Field

from unflow.core.unflow_core import unflowdecorator

# Define relevant variables for the ML task

def choose_optimizer(optimizer_name, model_parameters, **kwargs):
    if optimizer_name == "Adam":
        return torch.optim.Adam(model_parameters, **kwargs)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model_parameters, **kwargs)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(model_parameters, **kwargs)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(model_parameters, **kwargs)
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(model_parameters, **kwargs)
    elif optimizer_name == "Adadelta":
        return torch.optim.Adadelta(model_parameters, **kwargs)
    elif optimizer_name == "NAdam":
        return torch.optim.NAdam(model_parameters, **kwargs)
    elif optimizer_name == "ASGD":
        return torch.optim.ASGD(model_parameters, **kwargs)
    elif optimizer_name == "Muon":
        return torch.optim.Muon(model_parameters, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

class ModelConfig(BaseModel):
    num_classes: int = 10
    use_batchnorm: bool = True
    dropout_rate: float = 0.5
    n_blocks: int = 2
    block_channels: list = Field(default_factory=lambda: [6, 16])
    kernel_sizes: list = Field(default_factory=lambda: [5, 5])
    strides: list = Field(default_factory=lambda: [1, 1])
    paddings: list = Field(default_factory=lambda: [0, 0])
    n_fc_layers: int = 2
    fc_hidden_sizes: list = Field(default_factory=lambda: [120, 84])
    activation_fn: str = "ReLU"

class DataConfig(BaseModel):
    train_data_path: str = './data/train'
    test_data_path: str = './data/test'
    batch_size: int = 64


class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    num_epochs: int = 10
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_settings: ModelConfig = Field(default_factory=ModelConfig)
    data_settings: DataConfig = Field(default_factory=DataConfig)
    optimizer_params: dict = Field(default_factory=lambda: {'lr': 0.001})
    optimizer: str = "Adam"  # Default optimizer is Adam, can be changed to other optimizers like SGD, RMSprop, etc.


    


def set_loaders(data_config: DataConfig):
    # Load the MNIST dataset
    transforms_ = [transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    train_dataset = torchvision.datasets.MNIST(root=data_config.train_data_path,
                                               train=True,
                                               transform=transforms.Compose(transforms_),
                                               download=True)
    
    test_dataset = torchvision.datasets.MNIST(root=data_config.test_data_path,
                                              train=False,
                                              transform=transforms.Compose(transforms_),
                                              download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=data_config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=data_config.batch_size,
                                              shuffle=False)
    
    return train_loader, test_loader


#Defining the convolutional neural network
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_batchnorm=True, activation_fn="ReLU",
                 dropout_rate=0.5):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(getattr(nn, activation_fn)())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
def fc_block(in_features, out_features, use_batchnorm=True, activation_fn="ReLU", dropout_rate=0.5):
    layers = [nn.Linear(in_features, out_features)]
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_features))
    layers.append(getattr(nn, activation_fn)())
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)
class LeNet5(nn.Module):
    def __init__(self, num_classes, use_batchnorm=True, dropout_rate=0.5,activation_fn="ReLU", 
                 n_blocks=2, block_channels=[6, 16], kernel_sizes=[5, 5], strides=[1, 1], paddings=[0, 0],
                 n_fc_layers=2, fc_hidden_sizes=[120, 84]):
        super().__init__()
        self.conv_part = nn.Sequential()
        for i in range(n_blocks):
            in_channels = 1 if i == 0 else block_channels[i - 1]
            out_channels = block_channels[i]
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            self.conv_part.add_module(f"conv_block_{i+1}", 
                                      ConvBlock(in_channels, out_channels, kernel_size, stride, padding, use_batchnorm, activation_fn, dropout_rate))
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 32, 32)
            dummy_output = self.conv_part(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        self.fc_part = nn.Sequential()
        for i in range(n_fc_layers):
            in_features = flattened_size if i == 0 else fc_hidden_sizes[i - 1]
            out_features = fc_hidden_sizes[i]
            self.fc_part.add_module(f"fc_block_{i+1}", 
                                    fc_block(in_features, out_features, use_batchnorm, activation_fn, dropout_rate))
        self.fc = nn.Linear(fc_hidden_sizes[-1], num_classes)
       

    def forward(self, x):
        out = self.conv_part(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc_part(out)
        out = self.fc(out)
        return out
    
@unflowdecorator()
def run_experiment(training_config: TrainingConfig):
    # Set device
    device = torch.device(training_config.device)
    
    # Set data loaders
    train_loader, test_loader = set_loaders(training_config.data_settings)
    
    # Initialize the model
    model = LeNet5(num_classes=training_config.model_settings.num_classes,
                   use_batchnorm=training_config.model_settings.use_batchnorm,
                   dropout_rate=training_config.model_settings.dropout_rate,
                   n_blocks=training_config.model_settings.n_blocks,
                   block_channels=training_config.model_settings.block_channels,
                   kernel_sizes=training_config.model_settings.kernel_sizes,
                   strides=training_config.model_settings.strides,
                   paddings=training_config.model_settings.paddings,
                   n_fc_layers=training_config.model_settings.n_fc_layers,
                   fc_hidden_sizes=training_config.model_settings.fc_hidden_sizes).to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(training_config.optimizer, model.parameters(), **training_config.optimizer_params)

    def run_training_loop(model, train_loader, cost, optimizer, device, num_epochs):
        total_step = len(train_loader)
        model.train()  # Set the model to training mode
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):  
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = cost(outputs, labels)
                running_loss += loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sys.stdout.write(f'\rEpoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {running_loss/ (i+1):.4f}')
                sys.stdout.flush()
            print(f'\nEpoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/total_step:.4f}')
        return running_loss / total_step  # Return average loss for the last epoch
    def run_evaluation_loop(model, test_loader, device):
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')
            return accuracy
    loss = run_training_loop(model, train_loader, cost, optimizer, device, training_config.num_epochs)
    accuracy = run_evaluation_loop(model, test_loader, device)
    return {"loss": loss, "accuracy": accuracy}

