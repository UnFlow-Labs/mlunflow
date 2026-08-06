"""Test to see what actually triggers changes in PyTorch objects"""
import torch.nn as nn

from unflow.core.diff_args import get_args_changes

# Register a hook to modify the hooks dict
model = nn.Linear(10, 5)

args1 = {"model": model}
print("Initial vars keys:", list(vars(model).keys()))

# Register a forward hook
def dummy_hook(module, input, output):
    return output

model.register_forward_hook(dummy_hook)

args2 = {"model": model}

changes = get_args_changes(args1, args2)
print("\nAfter registering a hook:")
print(f"Changes detected: {bool(changes)}")
if changes:
    print(f"Change keys: {list(changes.keys())}")
    
# Check the hooks themselves
print(f"\nInitial _forward_hooks: {args1['model']._forward_hooks}")
print(f"Current _forward_hooks: {args2['model']._forward_hooks}")
print(f"_forward_hooks are same object: {args1['model']._forward_hooks is args2['model']._forward_hooks}")

# Also check the problematic _non_persistent_buffers_set
print(f"\n_non_persistent_buffers_set type: {type(model._non_persistent_buffers_set)}")
print(f"_non_persistent_buffers_set value: {model._non_persistent_buffers_set}")
