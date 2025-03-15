# Joint Gated Co-Attention Based Multi-Modal Networks for Subregion House Price Prediction

This project implements a multi-head gated co-attention mechanism for house price predictions using spatial data. The model leverages both geographic and euclidean distances between properties to create more accurate price estimations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Data Preparation](#data-preparation)
5. [Running the Model](#running-the-model)
6. [Architecture](#architecture)
7. [Model Parameters](#model-parameters)
8. [Results Interpretation](#results-interpretation)
9. [Troubleshooting](#troubleshooting)

## Project Overview

This project implements a neural network architecture that uses spatial information and multi-head gated co-attention to improve house price predictions. It works by:

1. Incorporating geographic relationships between properties
2. Using co-attention mechanisms to weigh the influence of neighboring properties
3. Learning embeddings that capture spatial patterns in housing markets
4. Combining multiple distance metrics (geographic and feature-based)

This approach typically outperforms traditional methods like kriging, IDW (Inverse Distance Weighting), and standard neural networks for spatial prediction tasks. The joint gated co-attention mechanism is particularly effective at capturing subregion-specific patterns in housing markets.

## Installation

### Prerequisites

- Python 3.8.10
- Conda
- CUDA-compatible GPU (optional but recommended)
- Graphviz (for visualization)

### Setup Environment

```bash
# Create and activate conda environment
conda create -n gca-house-price python=3.8.10
conda activate gca-house-price

# Install dependencies
pip install -r requirements.txt

# Install Graphviz for visualization (Windows)
# 1. Download from https://graphviz.gitlab.io/download/
# 2. Add to PATH environment variable
# 3. Install Python bindings
pip install pydot
```

## Project Structure

Based on the actual directory structure:

```
Project/
├── asi/                      # Core model implementation
│   ├── __pycache__/          # Python cache files
│   ├── __init__.py           # Module initialization
│   ├── attention_layer.py    # Primary attention mechanism
│   ├── attention_layer_2.py  # Secondary attention mechanism
│   ├── distance.py           # Distance calculations
│   ├── input_dataset.py      # Dataset processing
│   ├── input_neighborhood.py # Neighborhood definition handling
│   ├── input_phenomenon.py   # Phenomenon input processing
│   ├── interpolation.py      # Spatial interpolation methods
│   ├── model.py              # Main model architecture
│   └── transformation.py     # Data transformations
├── catboost_info/            # CatBoost model information
├── datasets/                 # Datasets organized by location
│   ├── BJ/                   # Beijing dataset
│   │   └── data.npz          # Preprocessed data
│   ├── IT/                   # Italy dataset
│   ├── kc/                   # King County dataset
│   └── poa/                  # Porto Alegre dataset
├── logs/                     # Training logs
├── notebooks/                # Jupyter notebooks for analysis
├── output/                   # Model outputs and visualizations
├── utils/                    # Utility functions
├── .gitignore                # Git ignore file
├── check_data.py             # Script to verify data structure
├── config.py                 # Configuration settings
├── create_input.py           # Data preprocessing script
└── README.md                 # Project documentation
```

## Data Preparation

The model expects preprocessed data in `.npz` format with specific arrays:

- `dist_eucli`: Euclidean distance matrix (N samples × K neighbors)
- `dist_geo`: Geographic distance matrix (N samples × K neighbors)
- `idx_eucli`: Indices of K nearest neighbors by Euclidean distance
- `idx_geo`: Indices of K nearest neighbors by geographic distance
- `X_train`, `X_test`: Feature matrices (including lat/long in first two columns)
- `y_train`, `y_test`: Target price values

### Checking Existing Data

You can verify your data structure using the check_data.py script:

```bash
python check_data.py
```

This will display all arrays in your dataset and their shapes, confirming they are properly formatted.

### Creating New Data

To process raw data into the required format:

```bash
python create_input.py datasets/your_dataset_folder/ 20
```

Where:
- First argument is the path to raw data folder
- Second argument is the number of nearest neighbors (K=20)

## Running the Model

### Option 1: With Visualization (requires Graphviz)

```bash
python run_training.py
```

### Option 2: Without Visualization

```bash
python run_training_no_viz.py
```

### Customizing the Model Configuration

Both scripts use a configuration dictionary that you can modify to adjust model parameters:

```python
model_config = {
    "id_dataset": "poa",           # Dataset to use
    "num_nearest_geo": 20,         # Neighbors (geographic)
    "num_nearest_eucli": 20,       # Neighbors (euclidean)
    "Num_heads": 4,                # Attention heads
    "learning_rate": 0.001,        # Learning rate
    "batch_size": 32,              # Batch size
    "num_neuron": 64,              # Hidden layer neurons
    "epochs": 100,                 # Training epochs
    "geointerpolation": "co_attention_multi", # Interpolation method
    # Add more parameters as needed
}
```

## Architecture

The Joint Gated Co-Attention model architecture consists of several key components:

### 1. Input Processing (`input_dataset.py`, `input_neighborhood.py`, `input_phenomenon.py`)
- Processes geographic coordinates (latitude, longitude)
- Handles property features
- Manages neighbor identification and relationships

### 2. Distance Calculation (`distance.py`)
- Computes geographic distances using haversine formula
- Calculates Euclidean distances in feature space
- Identifies K-nearest neighbors

### 3. Multi-Head Gated Co-Attention (`attention_layer.py`, `attention_layer_2.py`)
The core innovation of the model:
- Multiple attention heads learn different relationship patterns
- Co-attention mechanisms consider both geographic and feature spaces simultaneously
- Gating mechanism controls information flow between different attention channels
- Subregion-specific attention weights capture local market dynamics

### 4. Spatial Interpolation (`interpolation.py`)
- Implements various spatial interpolation techniques
- Applies attention-weighted spatial aggregation
- Handles the combination of geographic and feature influences

### 5. Transformation (`transformation.py`)
- Manages data transformations and normalization
- Converts between different distance/similarity representations

### 6. Model Integration (`model.py`)
- Combines all components into a cohesive architecture
- Manages the training and prediction pipeline
- Integrates feature embeddings with spatial representations

### Visual Architecture Flow

```
Input Data → Distance Calculation → Co-Attention Mechanism → Spatial Embedding → Output Layer
   ↑                                       ↑                          ↑
Feature Processing                  Multiple Heads               Transformation
```

## Model Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `sigma` | Gaussian kernel parameter | 0.1-1.0 |
| `learning_rate` | Model learning rate | 0.001-0.01 |
| `batch_size` | Training batch size | 32-128 |
| `num_neuron` | Neurons in hidden layers | 32-128 |
| `num_layers` | Number of hidden layers | 1-3 |
| `size_embedded` | Embedding dimension | 16-64 |
| `num_nearest_geo` | Geographic neighbors | 10-30 |
| `num_nearest_eucli` | Euclidean neighbors | 10-30 |
| `Num_heads` | Number of attention heads | 2-8 |
| `epochs` | Training epochs | 50-200 |
| `validation_split` | Validation data fraction | 0.1-0.2 |
| `geointerpolation` | Interpolation method | "co_attention_multi" |

## Results Interpretation

After training, the model reports several performance metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
- **RMSE (Root Mean Square Error)**: Square root of the average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage difference

Lower values indicate better performance for all metrics.

The model outputs and artifacts include:

1. **Console output**: Summary metrics for train and test sets
2. **Saved model**: Model weights and architecture
3. **Embedded representations**: Learned spatial patterns
4. **Attention weights**: Visualization of neighbor importance
5. **Model architecture diagram**: Visual representation of the network

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```
   ImportError: You must install pydot and graphviz
   ```
   **Solution**: Install Graphviz and pydot as described in the installation section, or use run_training_no_viz.py

2. **CUDA/GPU Issues**
   ```
   Could not load dynamic library 'cudnn64_8.dll'
   ```
   **Solution**: Install CUDA toolkit with cuDNN, or force CPU training by adding:
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
   ```

3. **Memory Issues**
   **Solution**: Reduce batch_size or num_nearest parameters

4. **Data Format Issues**
   **Solution**: Ensure data.npz contains all required arrays with correct shapes

### Performance Optimization

1. **Slow Training**:
   - Reduce number of neighbors
   - Reduce number of attention heads
   - Increase batch size (if memory allows)

2. **Poor Accuracy**:
   - Increase number of neighbors
   - Increase embedding dimensions
   - Add more hidden layers or neurons
   - Ensure features are properly normalized

3. **Overfitting**:
   - Reduce model complexity (fewer neurons, layers)
   - Add dropout layers
   - Use more training data

## Additional Resources

For more detailed information about the Joint Gated Co-Attention model architecture and attention mechanisms, refer to the codebase, particularly:

- `asi/model.py`: Core model implementation
- `asi/attention_layer.py` and `asi/attention_layer_2.py`: Attention mechanism details
- `asi/interpolation.py`: Spatial interpolation methods
- `asi/transformation.py`: Distance and similarity transformations

