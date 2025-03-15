import numpy as np

# Create a simple script to check data.npz structure
# Save this as check_data.py
data = np.load('datasets/poa/data.npz')
print("Available arrays in the dataset:")
for key in data.files:
    print(f"- {key}: {data[key].shape}")