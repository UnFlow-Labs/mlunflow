
# write a simple script that takes command line arguments and runs a procedure with those arguments
import argparse
from unflow.core.unflow_core import unflowdecorator, engine

@unflowdecorator()
def train_with_args(args):
    print(f"Training with learning rate: {args.lr}")
    # Simulate training process
    return {"status": "success"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a procedure with command line arguments.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")


    args = parser.parse_args()
    train_with_args(args)
    print(f"Graph edges after running the procedure: {engine.graph.graph.edges(data=True)}")