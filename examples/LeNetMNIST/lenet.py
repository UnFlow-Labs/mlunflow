# Load in relevant libraries, and alias where appropriate


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from pydantic import BaseModel, ConfigDict, Field

from unflow.core.unflow_core import unflowdecorator
from unflow.reasoning.detection import LLMIntentDetector
from unflow.reasoning.llm import LlamaLLM

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
    train_data_path: str = "./data/train"
    test_data_path: str = "./data/test"
    batch_size: int = 64


class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_epochs: int = 10
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    model_settings: ModelConfig = Field(default_factory=ModelConfig)
    data_settings: DataConfig = Field(default_factory=DataConfig)
    optimizer_params: dict = Field(default_factory=lambda: {"lr": 0.001})
    optimizer: str = "Adam"  # Default optimizer is Adam, can be changed to other optimizers like SGD, RMSprop, etc.


def set_loaders(data_config: DataConfig):
    # Load the MNIST dataset
    transforms_ = [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    train_dataset = torchvision.datasets.MNIST(
        root=data_config.train_data_path, train=True, transform=transforms.Compose(transforms_), download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_config.test_data_path, train=False, transform=transforms.Compose(transforms_), download=True
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=data_config.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=data_config.batch_size, shuffle=False)

    return train_loader, test_loader


# Defining the convolutional neural network
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        use_batchnorm=True,
        activation_fn="ReLU",
        dropout_rate=0.5,
    ):
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
    def __init__(
        self,
        num_classes,
        use_batchnorm=True,
        dropout_rate=0.5,
        activation_fn="ReLU",
        n_blocks=2,
        block_channels=[6, 16],
        kernel_sizes=[5, 5],
        strides=[1, 1],
        paddings=[0, 0],
        n_fc_layers=2,
        fc_hidden_sizes=[120, 84],
    ):
        super().__init__()
        self.conv_part = nn.Sequential()
        for i in range(n_blocks):
            in_channels = 1 if i == 0 else block_channels[i - 1]
            out_channels = block_channels[i]
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            self.conv_part.add_module(
                f"conv_block_{i + 1}",
                ConvBlock(
                    in_channels, out_channels, kernel_size, stride, padding, use_batchnorm, activation_fn, dropout_rate
                ),
            )
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 32, 32)
            dummy_output = self.conv_part(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.fc_part = nn.Sequential()
        for i in range(n_fc_layers):
            in_features = flattened_size if i == 0 else fc_hidden_sizes[i - 1]
            out_features = fc_hidden_sizes[i]
            self.fc_part.add_module(
                f"fc_block_{i + 1}", fc_block(in_features, out_features, use_batchnorm, activation_fn, dropout_rate)
            )
        self.fc = nn.Linear(fc_hidden_sizes[-1], num_classes)

    def forward(self, x):
        out = self.conv_part(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc_part(out)
        out = self.fc(out)
        return out

