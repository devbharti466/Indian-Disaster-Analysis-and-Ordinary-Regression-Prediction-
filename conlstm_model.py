"""
ConvLSTM Model for Indian Disaster Analysis and Prediction
===========================================================
This module implements a Convolutional LSTM (ConvLSTM) model for predicting
and analyzing disaster patterns across India using spatiotemporal data.

ConvLSTM captures both spatial (geographic) and temporal (time-series)
dependencies in disaster data, making it well-suited for:
 - Flood prediction
 - Cyclone/storm tracking
 - Drought forecasting
 - Earthquake aftershock prediction

Usage:
    python conlstm_model.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    ConvLSTM2D, BatchNormalization, Conv3D, Dense,
    Flatten, Dropout, Input, TimeDistributed, MaxPooling3D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Grid dimensions representing India divided into spatial cells
GRID_HEIGHT = 20          # Latitude grid cells (approx. 8°N–37°N)
GRID_WIDTH = 20           # Longitude grid cells (approx. 68°E–97°E)
N_CHANNELS = 3            # Feature channels: rainfall, temperature, disaster_index
SEQ_LEN = 12              # Look-back window: 12 time steps (e.g., 12 months)
PRED_LEN = 1              # Predict 1 step ahead
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data Generation (Synthetic Indian Disaster Data)
# ---------------------------------------------------------------------------

def generate_synthetic_disaster_data(n_samples: int = 500) -> np.ndarray:
    """
    Generate synthetic spatiotemporal disaster data representative of
    Indian geographic and seasonal patterns.

    The synthetic data encodes:
    - Channel 0: Normalised monthly rainfall (monsoon pattern)
    - Channel 1: Normalised temperature field
    - Channel 2: Disaster intensity index (floods, cyclones, droughts)

    Parameters
    ----------
    n_samples : int
        Number of time steps to generate.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, GRID_HEIGHT, GRID_WIDTH, N_CHANNELS).

    Performance note
    ----------------
    All channels are computed with fully vectorised NumPy operations across
    the time axis, avoiding a Python-level loop over timesteps.
    """
    data = np.zeros((n_samples, GRID_HEIGHT, GRID_WIDTH, N_CHANNELS))

    months = np.arange(n_samples) % 12  # shape: (n_samples,)

    # --- Rainfall channel (monsoon intensifies Jun–Sep) ---
    base_rainfall = np.random.rand(n_samples, GRID_HEIGHT, GRID_WIDTH)
    # Western Ghats and North-East receive more rain
    base_rainfall[:, 10:, :5] += 0.4   # Western Ghats
    base_rainfall[:, :5, 15:] += 0.3   # North-East India
    monsoon_factor = (
        0.5 * np.sin(2 * np.pi * (months - 3) / 12) + 0.5
    )[:, None, None]  # shape: (n_samples, 1, 1) for broadcasting
    data[:, :, :, 0] = np.clip(base_rainfall * monsoon_factor, 0, 1)

    # --- Temperature channel (hotter in summer) ---
    base_temp = np.random.rand(n_samples, GRID_HEIGHT, GRID_WIDTH)
    # Northern plains are hotter in summer
    base_temp[:, :10, :] += 0.2
    temp_factor = (
        0.5 * np.sin(2 * np.pi * (months - 1) / 12) + 0.5
    )[:, None, None]  # shape: (n_samples, 1, 1) for broadcasting
    data[:, :, :, 1] = np.clip(base_temp * temp_factor, 0, 1)

    # --- Disaster index (correlated with rainfall and temperature) ---
    disaster = (
        0.5 * data[:, :, :, 0]          # floods follow rainfall
        + 0.3 * data[:, :, :, 1]        # heat-wave index
        + 0.2 * np.random.rand(n_samples, GRID_HEIGHT, GRID_WIDTH)
    )
    # Coastal areas (bottom rows) hit by cyclones during Oct–Nov
    cyclone_months = np.isin(months, [9, 10])  # shape: (n_samples,)
    disaster[cyclone_months, -3:, :] += 0.4
    data[:, :, :, 2] = np.clip(disaster, 0, 1)

    return data


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------

def create_sequences(
    data: np.ndarray,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN
):
    """
    Slide a window over the time axis to create (X, y) pairs.

    Parameters
    ----------
    data : np.ndarray  shape (T, H, W, C)
    seq_len : int       input window length
    pred_len : int      prediction horizon (steps ahead)

    Returns
    -------
    X : np.ndarray  shape (N, seq_len, H, W, C)
    y : np.ndarray  shape (N, pred_len, H, W, C)

    Performance note
    ----------------
    Uses NumPy fancy indexing to build both arrays in two vectorised gather
    operations, avoiding a Python loop and repeated list appends.
    """
    n = len(data) - seq_len - pred_len + 1
    # Row indices for the input window: shape (n, seq_len)
    idx_x = np.arange(n)[:, None] + np.arange(seq_len)[None, :]
    # Row indices for the target window: shape (n, pred_len)
    idx_y = np.arange(n)[:, None] + seq_len + np.arange(pred_len)[None, :]
    X = data[idx_x]   # (n, seq_len, H, W, C)
    y = data[idx_y]   # (n, pred_len, H, W, C)
    return X, y


def prepare_dataset(data: np.ndarray, train_ratio: float = 0.8):
    """
    Normalise data and split into train / test sequence pairs.

    Parameters
    ----------
    data       : raw array (T, H, W, C)
    train_ratio: fraction of time steps used for training

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    scaler                            : fitted MinMaxScaler (per channel)
    """
    T, H, W, C = data.shape

    # Fit one scaler per channel across the spatial dimensions
    scalers = []
    scaled = np.zeros_like(data)
    for c in range(C):
        channel = data[:, :, :, c].reshape(T, -1)
        scaler = MinMaxScaler()
        scaled[:, :, :, c] = scaler.fit_transform(channel).reshape(T, H, W)
        scalers.append(scaler)

    X, y = create_sequences(scaled)

    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scalers


# ---------------------------------------------------------------------------
# ConvLSTM Model Architecture
# ---------------------------------------------------------------------------

def build_conlstm_model(
    input_shape: tuple,
    pred_len: int = PRED_LEN,
    filters: int = 64,
    kernel_size: tuple = (3, 3),
    dropout_rate: float = 0.2
) -> Model:
    """
    Build a ConvLSTM2D-based encoder–decoder model for spatiotemporal
    disaster prediction.

    Architecture
    ------------
    Input (seq_len, H, W, C)
     → ConvLSTM2D(filters,   return_sequences=True)  → BN → Dropout
     → ConvLSTM2D(filters*2, return_sequences=True)  → BN → Dropout
     → ConvLSTM2D(filters,   return_sequences=True)  → BN
     → Conv3D(C, sigmoid)  shape: (seq_len, H, W, C)
     → Lambda: last pred_len time-steps  shape: (pred_len, H, W, C)

    Only the last `pred_len` steps are produced as output so the training
    target never needs to be tiled to match the encoder sequence length.

    Parameters
    ----------
    input_shape  : (seq_len, H, W, C)
    pred_len     : number of future time steps to predict
    filters      : base number of convolutional filters
    kernel_size  : spatial kernel size for ConvLSTM2D
    dropout_rate : dropout probability

    Returns
    -------
    tf.keras.Model  output shape: (batch, pred_len, H, W, C)
    """
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    x = ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        name="convlstm_1"
    )(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = ConvLSTM2D(
        filters=filters * 2,
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        name="convlstm_2"
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        return_sequences=True,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        name="convlstm_3"
    )(x)
    x = BatchNormalization()(x)

    # --- Output head ---
    # Conv3D projects the full hidden sequence → channel space
    x = Conv3D(
        filters=input_shape[-1],  # N_CHANNELS
        kernel_size=(3, 3, 3),
        padding="same",
        activation="sigmoid",
        name="output_conv"
    )(x)  # shape: (batch, seq_len, H, W, C)

    # Slice the last pred_len time steps so output matches the target shape
    outputs = tf.keras.layers.Lambda(
        lambda t: t[:, -pred_len:, :, :, :],
        name="prediction_slice"
    )(x)  # shape: (batch, pred_len, H, W, C)

    model = Model(inputs=inputs, outputs=outputs, name="ConvLSTM_DisasterPredictor")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE
):
    """
    Compile and train the ConvLSTM model.

    The model output has shape (batch, pred_len, H, W, C) which matches
    y_train / y_val directly — no reshaping is required.

    Parameters
    ----------
    model      : compiled or uncompiled Keras Model
    X_train    : (N, seq_len, H, W, C)
    y_train    : (N, pred_len, H, W, C)
    X_val      : validation inputs
    y_val      : validation targets
    epochs     : training epochs
    batch_size : mini-batch size

    Returns
    -------
    history : Keras History object
    """
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            "best_conlstm_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple:
    """
    Evaluate the model on the test set and return regression metrics and
    predictions.

    Parameters
    ----------
    model  : trained Keras Model
    X_test : (N, seq_len, H, W, C)
    y_test : (N, pred_len, H, W, C)

    Returns
    -------
    metrics : dict with 'mse', 'rmse', 'mae', 'r2'
    y_pred  : np.ndarray  shape (N, pred_len, H, W, C)

    Performance note
    ----------------
    Returning ``y_pred`` avoids a redundant second ``model.predict`` call
    in the calling code when the predictions are needed for visualisation.
    """
    y_pred = model.predict(X_test, verbose=0)   # (N, pred_len, H, W, C)

    y_true_flat = y_test.reshape(len(y_test), -1)
    y_pred_flat = y_pred.reshape(len(y_pred), -1)

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k.upper():5s}: {v:.6f}")
    return metrics, y_pred


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_history(history) -> None:
    """Plot training and validation loss / MAE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("ConvLSTM Training History – Indian Disaster Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Training history saved to 'training_history.png'")


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int = 0,
    channel_names: list = None
) -> None:
    """
    Side-by-side spatial heatmaps of ground truth vs. predicted disaster maps.

    Parameters
    ----------
    y_true       : (N, pred_len, H, W, C)
    y_pred       : (N, pred_len, H, W, C)
    sample_idx   : which test sample to visualise
    channel_names: labels for each feature channel
    """
    if channel_names is None:
        channel_names = ["Rainfall", "Temperature", "Disaster Index"]

    n_channels = y_true.shape[-1]
    fig, axes = plt.subplots(n_channels, 2, figsize=(10, 4 * n_channels))

    for c in range(n_channels):
        true_map = y_true[sample_idx, -1, :, :, c]
        pred_map = y_pred[sample_idx, -1, :, :, c]

        vmin = min(true_map.min(), pred_map.min())
        vmax = max(true_map.max(), pred_map.max())

        im = axes[c, 0].imshow(true_map, cmap="hot", vmin=vmin, vmax=vmax)
        axes[c, 0].set_title(f"Ground Truth – {channel_names[c]}")
        axes[c, 0].axis("off")
        plt.colorbar(im, ax=axes[c, 0])

        im = axes[c, 1].imshow(pred_map, cmap="hot", vmin=vmin, vmax=vmax)
        axes[c, 1].set_title(f"Predicted – {channel_names[c]}")
        axes[c, 1].axis("off")
        plt.colorbar(im, ax=axes[c, 1])

    plt.suptitle(
        f"ConvLSTM Prediction vs Ground Truth (sample {sample_idx})",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig("prediction_comparison.png", dpi=150)
    plt.show()
    print("Prediction comparison saved to 'prediction_comparison.png'")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Indian Disaster Analysis – ConvLSTM Model")
    print("=" * 60)

    # 1. Generate / load data
    print("\n[1/5] Generating synthetic Indian disaster data …")
    data = generate_synthetic_disaster_data(n_samples=500)
    print(f"  Data shape: {data.shape}  (timesteps, H, W, channels)")

    # 2. Prepare sequences
    print("\n[2/5] Preparing train/test sequences …")
    X_train, X_test, y_train, y_test, scalers = prepare_dataset(data)
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape} | y_test: {y_test.shape}")

    # 3. Build model
    print("\n[3/5] Building ConvLSTM model …")
    input_shape = X_train.shape[1:]   # (seq_len, H, W, C)
    model = build_conlstm_model(input_shape)
    model.summary()

    # 4. Train
    print("\n[4/5] Training …")
    val_split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    history = train_model(model, X_tr, y_tr, X_val, y_val)
    plot_training_history(history)

    # 5. Evaluate and visualise
    # evaluate_model returns both metrics and predictions so that a second
    # model.predict call is not needed for the visualisation step.
    print("\n[5/5] Evaluating on test set …")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    plot_prediction_comparison(y_test, y_pred, sample_idx=0)

    print("\nDone! Model and plots saved.")
    return model, history, metrics


if __name__ == "__main__":
    main()
