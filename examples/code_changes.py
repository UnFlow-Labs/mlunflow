from unflow.core.unflow_core import unflowdecorator


tracker = unflowdecorator()


@tracker
def train_model(model: str, dataset: str, lr: float):
    print(f"Baseline training for {model} on {dataset} with lr={lr}")
    return {"accuracy": 0.81}


train_model.clear_graph()

print("\nRun 1 (baseline code):")
train_model(model="resnet18", dataset="cifar10", lr=0.001)


@tracker
def train_model(model: str, dataset: str, lr: float):
    print(f"Updated training for {model} on {dataset} with lr={lr}")
    print("Applied data augmentation and weight decay.")
    return {"accuracy": 0.86}


print("\nRun 2 (updated code, same args):")
train_model(model="resnet18", dataset="cifar10", lr=0.001)


graph = train_model.compute_graph.graph
print(f"\nGraph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

for source, target, data in graph.edges(data=True):
    transformation = data["transformation"]
    print(f"\nTransformation: {source} -> {target}")
    print(f"Args changes: {transformation.args_changes}")
    if transformation.p_changes:
        print("Procedure/code changes detected:")
        for _, diff_line in transformation.p_changes.items():
            print(diff_line)
