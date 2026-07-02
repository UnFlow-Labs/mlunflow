from unflow.core.unflow_core import unflowdecorator


@unflowdecorator()
def _train(lr, epochs, optimizer="adam", batch_size=32, model="resnet", dataset="cifar10", fail=False):
    print(
        f"Training with learning rate: {lr}, epochs: {epochs}, optimizer: {optimizer},"
        f" batch size: {batch_size}, model: {model}, dataset: {dataset}"
    )
    if fail:
        print("Simulating failure for this run.")
        raise Exception("Simulated failure")
    return {"loss": 0.5}


def test_clear_graph():
    _train.clear_graph()
    assert _train.graph_size() == 0


def test_multiple_runs():
    combinations = [
        {"lr": 0.01, "epochs": 10, "optimizer": "adam", "batch_size": 32, "model": "resnet", "dataset": "cifar10"},
        {"lr": 0.001, "epochs": 20, "optimizer": "sgd", "batch_size": 64, "model": "vgg16", "dataset": "imagenet"},
        {
            "lr": 0.05,
            "epochs": 15,
            "optimizer": "rmsprop",
            "batch_size": 128,
            "model": "mobilenet",
            "dataset": "cifar100",
        },
    ]

    for combo in combinations:
        result = _train(**combo)
        assert result == {"loss": 0.5}


def test_adding_an_executed_node():
    result1 = _train(lr=0.01, epochs=10)
    assert result1 is None

    result2 = _train(lr=0.001, epochs=20)
    assert result2 == {"loss": 0.5}

    # Check that the graph has two nodes
    assert _train.graph_size() == 4


def test_adding_adding_a_node_depending_on_another_failed_node():
    result1 = _train(lr=0.01, epochs=10, optimizer="rmsprop", fail=True)
    assert result1 is None
    result2 = _train(lr=0.001, epochs=20, optimizer="rmsprop", fail=False)
    assert result2 is None

    # Check that the graph has two nodes
    assert _train.graph_size() == 6
