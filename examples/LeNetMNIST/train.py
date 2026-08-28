import sys
import torch
import torch.nn as nn
from unflow.core.unflow_core import unflowdecorator
from unflow.reasoning.detection import LLMIntentDetector
from unflow.reasoning.llm import LlamaLLM

from examples.LeNetMNIST.lenet import LeNet5, set_loaders, choose_optimizer
from examples.LeNetMNIST.lenet import TrainingConfig
from pathlib import Path



intent_detector = LLMIntentDetector(llm=LlamaLLM(repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", 
        filename="qwen2.5-coder-7b-instruct-q8_0.gguf", n_ctx=4096))

@unflowdecorator(intent_detector= intent_detector, context_files=[Path(__file__).parent / "lenet.py"])
def run_experiment(training_config: TrainingConfig):
    print(f"Running experiment with config: {training_config}")
    # Set device
    device = torch.device(training_config.device)
   
    # Set data loaders
    train_loader, test_loader = set_loaders(training_config.data_settings)

    # Initialize the model
    model = LeNet5(
        num_classes=training_config.model_settings.num_classes,
        use_batchnorm=training_config.model_settings.use_batchnorm,
        dropout_rate=training_config.model_settings.dropout_rate,
        n_blocks=training_config.model_settings.n_blocks,
        block_channels=training_config.model_settings.block_channels,
        kernel_sizes=training_config.model_settings.kernel_sizes,
        strides=training_config.model_settings.strides,
        paddings=training_config.model_settings.paddings,
        n_fc_layers=training_config.model_settings.n_fc_layers,
        fc_hidden_sizes=training_config.model_settings.fc_hidden_sizes,
    ).to(device)
   
    cost = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(training_config.optimizer, model.parameters(), **training_config.optimizer_params)

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
                print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f} %")
                return accuracy

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
                sys.stdout.write(
                    f"\rEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {running_loss / (i + 1):.4f}"
                )
                sys.stdout.flush()
            accuracy = run_evaluation_loop(model, test_loader, device)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / total_step:.4f}, Accuracy: {accuracy:.2f}%")
        return running_loss / total_step  # Return average loss for the last epoch

   

    loss = run_training_loop(model, train_loader, cost, optimizer, device, training_config.num_epochs)
    accuracy = run_evaluation_loop(model, test_loader, device)
    return {"loss": loss, "accuracy": accuracy}
