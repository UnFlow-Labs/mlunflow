import typer

from unflow.core.unflow_core import c_graph, unflowdecorator

app = typer.Typer()


@unflowdecorator
def train(lr=0.01, epochs=10, optimizer="adam", batch_size=32, model="resnet", dataset="cifar10"):
    # print(f"Training with learning rate: {lr} and epochs: {epochs}")
    epochs = epochs + 10
    return {"epochs": epochs}


if __name__ == "__main__":
    # train(lr=0.001, epochs=5, optimizer="sgd", batch_size=64, model="vgg2", dataset="mnist")
    train(lr=0.01, epochs=100, optimizer="", batch_size=1, model="resnet", dataset="cifar10")
    train(lr=0.005, epochs=15, optimizer="rmsprop", batch_size=128, model="mobilenet")
    # train(lr=0.005, epochs=15, optimizer="lion", batch_size=128, model="mobilenet")

    edges = c_graph.graph.edges(data=True)
    for edge in edges:
        print(f"Edge from {edge[0]} to {edge[1]} with transformation: {edge[2]['transformation'].name}")
