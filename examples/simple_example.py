from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def train_2(lr, epochs, optimizer, batch_size, model, dataset):
    print(f"Training {model} on {dataset} with lr={lr}, epochs={epochs}, optimizer={optimizer}, batch_size={batch_size}")
    # Here you would add the actual training logic, e.g., loading data, defining the model, training loop, etc.
    # For demonstration purposes, we just print the parameters.


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
    train_2(**combinations[0])
    train_2(**combinations[1])
    train_2(**combinations[2])
    train_2(**combinations[3])
    # print(f"Graph edges after running the procedure: {engine.graph.graph.edges(data=True)}")
