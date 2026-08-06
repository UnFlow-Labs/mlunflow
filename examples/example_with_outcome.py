from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def train_2(lr, epochs=10, optimizer="adam", batch_size=32, model="resnet", dataset="cifar10"):
    print(
        f"Training with learning rate: {lr}, epochs: {epochs}, optimizer: {optimizer}, "
        f"batch size: {batch_size}, model: {model}, dataset: {dataset}"
    )
    # Simulate training process
    print("Training completed.")
    return {
        "lr": lr,
        "epochs": epochs,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "model": model,
        "dataset": dataset,
        "accuracy": 0.9,
    }


if __name__ == "__main__":
    combinations = [
        {"lr": 0.01, "epochs": 0, "optimizer": "adam", "batch_size": 32, "model": "resnet", "dataset": "cifar10"},
        {"lr": 0.001, "epochs": 20, "optimizer": "sgd", "batch_size": 64, "model": "vgg16", "dataset": "imagenet"},
        {
            "lr": 0.05,
            "epochs": 15,
            "optimizer": "rmsprop",
            "batch_size": 128,
            "model": "mobilenet",
            "dataset": "cifar100",
        },
        {
            "lr": 505050000000,
            "epochs": 100,
            "optimizer": "rmsprop",
            "batch_size": 128,
            "model": "mobilenet",
            "dataset": "cifar100",
        },
    ]
    # train_2.clear_graph()
    for combo in combinations:
        train_2(**combo)
        outcomes = train_2.get_outcomes()
for state_name, outcome in outcomes.items():
    print(f"State: {state_name}, Outcome: {outcome.to_json()}")
