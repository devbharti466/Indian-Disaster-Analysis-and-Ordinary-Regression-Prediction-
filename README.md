# Indian Disaster Analysis and Regression Prediction

Spatiotemporal analysis and prediction of natural disasters across India using deep learning.

## Models

### ConvLSTM (Convolutional LSTM)
`conlstm_model.py` implements a **ConvLSTM2D** encoder–decoder that captures both:
- **Spatial** patterns – geographic spread of floods, cyclones, droughts
- **Temporal** patterns – seasonal and multi-year disaster cycles

#### Architecture
```
Input (seq_len, H, W, C)
 └─► ConvLSTM2D(32)  → BatchNorm → Dropout
 └─► ConvLSTM2D(64)  → BatchNorm → Dropout
 └─► ConvLSTM2D(32)  → BatchNorm
 └─► Conv3D(C, sigmoid)   ← output: predicted spatial map
```

#### Feature Channels
| Channel | Feature | Description |
|---------|---------|-------------|
| 0 | Rainfall | Monthly normalised rainfall (monsoon pattern) |
| 1 | Temperature | Monthly normalised temperature field |
| 2 | Disaster Index | Combined flood / cyclone / drought intensity |

#### Spatial Grid
India is divided into a **20 × 20 grid** (lat 8°N–37°N, lon 68°E–97°E).

## Quick Start

```bash
pip install tensorflow scikit-learn matplotlib numpy
python conlstm_model.py
```

Or open the interactive notebook:
```bash
jupyter notebook indian_disaster_conlstm.ipynb
```

## Files
| File | Description |
|------|-------------|
| `conlstm_model.py` | ConvLSTM model, data generation, training & evaluation |
| `indian_disaster_conlstm.ipynb` | Step-by-step Jupyter notebook walkthrough |