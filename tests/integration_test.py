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
    _train.clear_graph()
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
        assert result is not None
        assert result == {"loss": 0.5}


def test_adding_an_executed_node():
    _train.clear_graph()
    result1 = _train(lr=0.01, epochs=10)
    assert result1 is not None
    assert result1 == {"loss": 0.5}

    duplicate = _train(lr=0.01, epochs=10)
    assert duplicate is None

    result2 = _train(lr=0.001, epochs=20)
    assert result2 is not None
    assert result2 == {"loss": 0.5}

    # Check that the graph has two nodes
    assert _train.graph_size() == 2


def test_adding_adding_a_node_depending_on_another_failed_node():
    _train.clear_graph()
    result1 = _train(lr=0.01, epochs=10, optimizer="rmsprop", fail=True)
    assert result1 is None
    result2 = _train(lr=0.001, epochs=20, optimizer="rmsprop", fail=False)
    assert result2 is None

    # Check that the graph has two nodes
    assert _train.graph_size() == 2


def test_outcomes():
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
    _train.clear_graph()
    for combo in combinations:
        result = _train(**combo)
        assert result is not None
    outcomes = _train.get_outcomes()
    assert len(outcomes) == len(combinations)
    for outcome in outcomes.values():
        assert outcome is not None
        assert outcome.outputs == {"loss": 0.5}
