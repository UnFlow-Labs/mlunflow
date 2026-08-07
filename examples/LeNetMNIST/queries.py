from examples.LeNetMNIST.lenet import run_experiment


def get_change_field(change, field_name):
    if isinstance(change, dict):
        return change.get(field_name)
    return getattr(change, field_name, None)


def activation_change_exists(transformation):
    if "activation_fn" not in transformation.args_changes:
        return False
    if len(transformation.args_changes) > 1:
        return False
    # if "0" in transformation.state1.name:
    #     return False
    change = transformation.args_changes["activation_fn"]
    return any(
        get_change_field(change, field_name) is not None and change["to_value"] == "Tanh"
        for field_name in ("from_value", "to_value", "from_type", "to_type")
    )


changes = run_experiment.query_transformations(predicate=activation_change_exists)
print(f"Found {len(changes)} transformations that changed the activation function.")
outcomes = run_experiment.get_outcomes()
for transformation in changes:
    # for arg, change in transformation.args_changes.items():
    #     from_value = get_change_field(change, "from_value")
    #     to_value = get_change_field(change, "to_value")
    #     from_type = get_change_field(change, "from_type")
    #     to_type = get_change_field(change, "to_type")

    #     if from_value is not None or to_value is not None:
    #         print(f"- {arg}: from {from_value} to {to_value}")
    #     elif from_type is not None or to_type is not None:
    #         print(f"- {arg}:")
    # else:
    #     print(f"- {arg}: change details not available")
    # get the source and target states
    source_state = transformation.state1
    target_state = transformation.state2
    # get the outcome of source and target states
    try:
        source_outcome = outcomes.get(source_state.name).outputs
        target_outcome = outcomes.get(target_state.name).outputs
    except Exception:
        # print(f"Error retrieving outcomes for states {source_state.name} and {target_state.name}: {e}")
        continue
    if source_outcome is None or target_outcome is None:
        # print(f"Outcomes not available for states {source_state.name} and/or {target_state.name}.")
        continue

    change_in_accuracy = target_outcome["accuracy"] - source_outcome["accuracy"]
    if change_in_accuracy > 0:
        print(f"Transformation from {source_state.name} to {target_state.name}:")
        print(
            f"  - Activation function changed from {transformation.args_changes['activation_fn'].get('from_value')} to {transformation.args_changes['activation_fn'].get('to_value')}"
        )
        print(
            f"Change in accuracy: +{change_in_accuracy:.4f} from {source_outcome['accuracy']:.2f}% to {target_outcome['accuracy']:.2f}%"
        )
    else:
        print(f"Transformation from {source_state.name} to {target_state.name}:")
        print(
            f"  - Activation function changed from {transformation.args_changes['activation_fn'].get('from_value')} to {transformation.args_changes['activation_fn'].get('to_value')}"
        )
        print(
            f"Change in accuracy: {change_in_accuracy:.4f} from {source_outcome['accuracy']:.2f}% to {target_outcome['accuracy']:.2f}%"
        )
