"""Test comparing two separately created models with hooks"""

import torch.nn as nn

from unflow.core.diff_args import _to_comparable, get_args_changes

# Create two identical models separately
model1 = nn.Linear(10, 5)
model2 = nn.Linear(10, 5)


# Register the same hook on both
def dummy_hook(module, input, output):
    return output


model1.register_forward_hook(dummy_hook)
model2.register_forward_hook(dummy_hook)

args1 = {"model": model1}
args2 = {"model": model2}

changes = get_args_changes(args1, args2)
print("Comparing two separately created models with hooks:")
print(f"Changes detected: {bool(changes)}")
if changes:
    print(f"Change keys: {list(changes.keys())}")
    for key in changes:
        print(f"  {key}")

# Let's manually check what gets compared
comparable1 = _to_comparable(args1)
comparable2 = _to_comparable(args2)
print(f"\nComparable args1 model keys: {list(comparable1['model'].keys())}")
print(f"Comparable args2 model keys: {list(comparable2['model'].keys())}")

# Check the _forward_hooks specifically
print(f"\nmodel1._forward_hooks id: {id(model1._forward_hooks)}")
print(f"model2._forward_hooks id: {id(model2._forward_hooks)}")
print(f"model1._forward_hooks == model2._forward_hooks: {model1._forward_hooks == model2._forward_hooks}")
print(f"model1._forward_hooks[0] is model2._forward_hooks[0]: {model1._forward_hooks[0] is model2._forward_hooks[0]}")

# Check what happens in comparable dicts
print(f"\ncomparable1['model']['_forward_hooks'] id: {id(comparable1['model']['_forward_hooks'])}")
print(f"comparable2['model']['_forward_hooks'] id: {id(comparable2['model']['_forward_hooks'])}")
