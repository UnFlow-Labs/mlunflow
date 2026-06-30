
# write a simple script that takes command line arguments and runs a procedure with those arguments
import argparse
from unflow.core.unflow_core import unflowdecorator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a procedure with command line arguments.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model", type=str, default="resnet", help="Model type")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")

    args = parser.parse_args()

    @unflowdecorator
    def train(lr=args.lr, epochs=args.epochs, optimizer=args.optimizer, batch_size=args.batch_size, model=args.model, dataset=args.dataset):
        print(f"Training with learning rate: {lr}, epochs: {epochs}, optimizer: {optimizer}, batch size: {batch_size}, model: {model}, dataset: {dataset}")
        # Simulate training process
        return {"status": "success"}

    train()