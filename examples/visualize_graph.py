from unflow.core.unflow_core import unflowdecorator

wer = {
    "1": [10, 4, 5, 16],
    "2": [7, 8, 9, 12],
    "1_h": [10, 4, 5, 16],
    "2_h": [7, 8, 9, 12],
}


@unflowdecorator()
def evaluate_function(model, data, hesitation=False):
    # Simulate evaluation process
    if hesitation:
        if data == "1":
            return wer["1_h"][model]
        elif data == "2":
            return wer["2_h"][model]
    else:
        if data == "1":
            return wer["1"][model]
        elif data == "2":
            return wer["2"][model]


evaluate_function.clear_graph()
models = [0, 1, 2, 3]
data_sets = ["1", "2"]
for model in models:
    for data in data_sets:
        evaluate_function(model, data, hesitation=False)
        evaluate_function(model, data, hesitation=True)

G = evaluate_function.compute_graph.graph
print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

states_for_model_0 = evaluate_function.query_states(args_contains={"model": 0})
print("\nStates for model=0:")
for state in states_for_model_0:
    print(f"- {state.name}: {state.args}")

hesitation_changes = evaluate_function.query_transformations(
    predicate=lambda transformation: (
        "hesitation" in transformation.args_changes
        and transformation.args_changes["hesitation"]["to_value"]
        != transformation.args_changes["hesitation"]["from_value"]
    )
)
print("\nTransformations that changed 'hesitation':")
for transformation in hesitation_changes:
    print(f"- {transformation.state1.name} -> {transformation.state2.name}: {transformation.args_changes}")

path = evaluate_function.shortest_path("evaluate_function_0", "evaluate_function_1")
print("\nShortest path evaluate_function_0 -> evaluate_function_1:")
print(" -> ".join(state.name for state in path))
