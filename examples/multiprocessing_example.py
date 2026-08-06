import os

from unflow.core.executors.multiprocessing_executor import MultiprocessingExecutor
from unflow.core.unflow_core import unflowdecorator


@unflowdecorator(executor=MultiprocessingExecutor())
def train_2(lr, epochs=10):
    for _i in range(1000000000):
        pass
    print(os.getpid())
    print(f"Training with learning rate: {lr}, epochs: {epochs}")

    # Simulate training process
    return {"status": "success"}


if __name__ == "__main__":
    combinations = [
        {"lr": 0.01, "epochs": 100},
        {"lr": 0.001, "epochs": 200},
    ]
    # train_2.clear_graph()
    train_2.run_multiple(combinations)
