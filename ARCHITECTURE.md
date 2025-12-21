# Model Architecture

## Overview

STA-UKAN is a deep learning architecture designed for subseasonal temperature forecasting. The model integrates multiple atmospheric factors with terrain information through a novel combination of:

1. **Multisource Atmospheric Factor Fusion**
2. **Terrain-Aware Token-KAN (Kolmogorov-Arnold Network)**
3. **Forecast Generation Head**

## Architecture Details

### 1. Input Layer

**Atmospheric Factors:**
- Multiple time series inputs (temperature, pressure, humidity, wind, etc.)
- Shape: `(batch_size, sequence_length, feature_dim)` for each factor
- Typical sequence length: 30 days of historical data

**Terrain Features:**
- Static spatial features (elevation, slope, aspect, etc.)
- Shape: `(batch_size, terrain_dim)`

### 2. Multisource Fusion Module

This module intelligently combines multiple atmospheric factors:

**Factor Encoders:**
- Individual encoders for each atmospheric factor
- 2-layer MLP with LayerNorm and GELU activation
- Projects each factor to common embedding dimension

**Cross-Factor Attention:**
- Multi-head attention between different factors
- Allows information exchange and correlation learning
- Multiple fusion layers for deep interaction

**Output:**
- Unified atmospheric representation
- Shape: `(batch_size, sequence_length, embed_dim)`

### 3. Terrain-Aware Token-KAN

The core innovation combining KAN with terrain awareness:

**Terrain Encoder:**
- 2-layer MLP for terrain feature encoding
- Projects terrain features to embedding dimension

**Token-KAN Blocks:**
Each block contains:

a. **Multi-Head Self-Attention:**
   - Captures temporal dependencies
   - Standard transformer-style attention
   
b. **Terrain Modulation:**
   - Terrain embedding modulates attention output
   - Uses sigmoid gating mechanism
   
c. **KAN Transformation:**
   - Kolmogorov-Arnold Network layers
   - B-spline basis functions for feature transformation
   - Non-linear function approximation

**KAN Layer Details:**
- Uses learnable B-spline basis functions
- Grid-based function representation
- Spline order: 3 (cubic splines)
- Grid size: 5 control points

### 4. Temporal Aggregation

- Mean pooling over sequence dimension
- Additional MLP transformation
- Produces fixed-size representation

### 5. Forecast Head

- 2-layer MLP with dropout
- Projects to forecast horizon (typically 14 days)
- Output shape: `(batch_size, forecast_horizon)`

## Kolmogorov-Arnold Network (KAN)

### Theory

Based on the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be represented as a composition of continuous univariate functions.

### Implementation

**B-Spline Basis:**
```
φ_i(x) = B-spline basis functions on grid
Output = Σ c_i * φ_i(x)
```

**Advantages:**
- More expressive than traditional MLPs
- Better function approximation
- Learnable activation functions

### KAN Layer Components

1. **Grid Points:** Uniformly distributed control points
2. **Coefficients:** Learnable spline coefficients
3. **Basis Computation:** B-spline basis evaluation
4. **Output:** Weighted sum of basis functions

## Model Flow

```
Input: {factor_1, factor_2, ..., factor_n}, terrain

1. Factor Encoding
   factor_i → Encoder_i → encoded_i

2. Cross-Factor Fusion
   {encoded_1, ..., encoded_n} → Fusion → fused_features

3. Terrain Encoding
   terrain → TerrainEncoder → terrain_embed

4. Token-KAN Processing
   for each Token-KAN block:
      - Multi-head attention
      - Terrain modulation
      - KAN transformation
      - Residual connection & LayerNorm

5. Temporal Aggregation
   fused_features → Mean Pool → pooled

6. Forecast Generation
   pooled → Forecast Head → predictions
```

## Key Features

### Attention Mechanisms

**Self-Attention:**
- Captures temporal patterns in atmospheric data
- Multi-head design for diverse representations

**Cross-Attention:**
- Enables factor interaction
- Learns correlations between different atmospheric variables

### Terrain Awareness

- Static terrain features modulate dynamic atmospheric patterns
- Spatial conditioning through multiplicative gating
- Helps model adapt predictions to local geography

### Residual Connections

- Throughout the architecture
- Enables deep networks (4+ Token-KAN layers)
- Stabilizes training

### Normalization

- LayerNorm after each major component
- Batch normalization avoided (better for variable-length sequences)

## Training Details

### Loss Function
- Mean Squared Error (MSE) for temperature prediction
- Can be extended to other loss functions

### Optimization
- AdamW optimizer (Adam with weight decay)
- Cosine annealing learning rate schedule
- Gradient clipping (max norm = 1.0)

### Regularization
- Dropout in Token-KAN blocks and forecast head
- Weight decay in optimizer
- Layer normalization

## Hyperparameters

### Model Configuration
- Embedding dimension: 256
- Number of attention heads: 8
- Number of fusion layers: 2
- Number of Token-KAN layers: 4
- KAN hidden dimensions: [512, 256]
- Dropout rate: 0.1

### Training Configuration
- Batch size: 32
- Learning rate: 0.001
- Weight decay: 0.0001
- Number of epochs: 100

## Performance Characteristics

### Computational Complexity

**Time Complexity:**
- Self-Attention: O(L² × D) where L = sequence length, D = embed dim
- KAN Layer: O(L × D × G) where G = grid size
- Overall: O(L² × D + L × D²)

**Space Complexity:**
- Parameters: ~2-10M depending on configuration
- Memory: O(B × L × D) where B = batch size

### Scalability

- Efficient for moderate sequence lengths (L ≤ 100)
- Parallelizable across GPUs
- Can handle multiple atmospheric factors

## Extensions and Variations

### Possible Modifications

1. **Additional Factors:** Add more atmospheric variables
2. **Hierarchical Attention:** Multi-scale temporal modeling
3. **Uncertainty Quantification:** Probabilistic outputs
4. **Transfer Learning:** Pre-training on multiple regions
5. **Multi-Task Learning:** Forecast multiple variables

### Advanced Features

- **Attention Visualization:** Analyze what the model focuses on
- **Factor Importance:** Measure contribution of each factor
- **Ablation Studies:** Test component importance
