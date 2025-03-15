from train_model import train

# Select which dataset to use from your directory
# Options appear to be: "BJ", "IT", "kc", "poa"
DATASET = "poa"  # Change this to the desired dataset

# Configure model parameters
model_config = {
    "sigma": 0.5,                 # Gaussian kernel parameter
    "learning_rate": 0.001,       # Learning rate
    "batch_size": 32,             # Batch size
    "num_neuron": 64,             # Number of neurons
    "num_layers": 2,              # Number of layers
    "size_embedded": 32,          # Size of embedding
    "num_nearest_geo": 20,        # Number of nearest neighbors (geo)
    "num_nearest_eucli": 20,      # Number of nearest neighbors (euclidean)
    "id_dataset": DATASET,        # Dataset ID
    "label": f"model_{DATASET}",  # Model label
    "graph_label": f"g_{DATASET}",# Graph label
    "num_nearest": 20,            # Number of nearest neighbors
    "geointerpolation": "asi_multi", # Interpolation method (uses multi-head attention)
    "Num_heads": 4,               # Number of attention heads
    "epochs": 100,                # Number of epochs
    "validation_split": 0.1,      # Validation split
    "early_stopping": True,       # Early stopping
    "scale_log": True,            # Scale log
    "optimier": "adam",           # Optimizer
    "type_compat_funct_eucli": "kernel_gaussiano", # Compatibility function
}

# Train the model
model = train(**model_config)
spatial, result, fit, embedded_train, embedded_test, pred_train, pred_test = model()

# Print results
print(f"\nResults for dataset: {DATASET}")
print(f"MAE (test): {result[0]}")
print(f"RMSE (test): {result[1]}")
print(f"MAPE (test): {result[2]}%")
print(f"MAE (train): {result[3]}")
print(f"RMSE (train): {result[4]}")
print(f"MAPE (train): {result[5]}%")