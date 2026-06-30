from unflow.core.unflow_core import unflowdecorator

@unflowdecorator()
def train(lr=0.01, epochs=10, optimizer="adam", batch_size=32, model="resnet", dataset="cifar10"):
    print(f"Training with learning rate: {lr}, epochs: {epochs}, optimizer: {optimizer}, batch size: {batch_size}, model: {model}, dataset: {dataset}")
    # Simulate training process
    return {"status": "success"}
if __name__ == "__main__":
    combinations = [
        {"lr": 0.01, "epochs": 10, "optimizer": "adam", "batch_size": 32, "model": "resnet", "dataset": "cifar10"},
        {"lr": 0.001, "epochs": 20, "optimizer": "sgd", "batch_size": 64, "model": "vgg16", "dataset": "imagenet"},
        {"lr": 0.005, "epochs": 15, "optimizer": "rmsprop", "batch_size": 128, "model": "mobilenet", "dataset": "cifar100"},
    ]
    results = train.run_in_parallel(combinations)
    print(f"Results: {results}")