import sys
from pathlib import Path

# Add the project root to sys.path to resolve imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.LeNetMNIST.lenet import DataConfig, ModelConfig, TrainingConfig
from examples.LeNetMNIST.train import run_experiment


run_experiment.clear_graph()  # Clear the graph before starting the experiment
 # Print the graph before starting the experiment
training_config = TrainingConfig(num_epochs=5, optimizer="Adam", optimizer_params={"lr": 0.001})
result = run_experiment(training_config)
print(f"Final Result: Loss = {result['loss']:.4f}, Accuracy = {result['accuracy']:.2f}%")
print(run_experiment.print_graph()) 
