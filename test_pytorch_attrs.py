"""Test to demonstrate the PyTorch attributes issue with model mutation"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from unflow.core.diff_args import get_args_changes

# Test 1: Same model instance before and after forward pass
model = nn.Linear(10, 5)
x = torch.randn(3, 10)

args1 = {"model": model, "lr": 0.001}
# Simulate model being used (gradient tracking might change internal state)
_ = model(x)
args2 = {"model": model, "lr": 0.001}

changes = get_args_changes(args1, args2)
print("Test 1 - Same model instance before/after forward pass:")
print(f"Changes detected: {bool(changes)}")
if changes:
    print(f"Change keys: {list(changes.keys())}")

# Test 2: With BatchNorm (which has running stats that change)
model_bn = nn.Sequential(nn.Linear(10, 5), nn.BatchNorm1d(5))
model_bn.train()

args1 = {"model": model_bn, "lr": 0.001}
x = torch.randn(3, 10)
_ = model_bn(x)  # This updates running mean/std
args2 = {"model": model_bn, "lr": 0.001}

changes = get_args_changes(args1, args2)
print("\nTest 2 - Model with BatchNorm after forward pass:")
print(f"Changes detected: {bool(changes)}")
if changes:
    print(f"Change keys: {list(changes.keys())}")
    for key in changes:
        print(f"  {key}")

# Test 3: Let's check what's in vars()
model_simple = nn.Linear(10, 5)
print("\nTest 3 - Checking vars() on PyTorch model:")
vars_dict = vars(model_simple)
print(f"Keys in vars(nn.Linear): {list(vars_dict.keys())}")

# Test 4: Check transform vars
transform = transforms.Normalize((0.1307,), (0.3081,))
print("\nTest 4 - Checking vars() on transforms.Normalize:")
vars_dict = vars(transform)
print(f"Keys in vars(Normalize): {list(vars_dict.keys())}")
