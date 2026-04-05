"""
backend/models.py
Model builders for ANN, CNN, and RNN.
Each function returns a compiled-ready (but not yet compiled) Keras model.
All architectural hyperparameters are explicit keyword arguments so the
frontend can expose every knob directly.
"""

import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _l1l2(l1: float, l2: float):
    if l1 == 0 and l2 == 0:
        return None
    return regularizers.L1L2(l1=l1, l2=l2)


def get_model_summary(model) -> str:
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()


def count_params(model) -> dict:
    total      = model.count_params()
    trainable  = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    return {"total": total, "trainable": trainable,
            "non_trainable": total - trainable}


# ─────────────────────────────────────────────────────────────────
# ANN  –  Dense feedforward classifier
# ─────────────────────────────────────────────────────────────────

def build_ann(
    *,
    input_dim:      int,
    n_classes:      int,
    hidden_layers:  int   = 2,
    units:          int   = 64,
    activation:     str   = "relu",
    dropout_rate:   float = 0.0,
    l1_lambda:      float = 0.0,
    l2_lambda:      float = 0.0,
    use_batch_norm: bool  = False,
    use_skip_conn:  bool  = False,
) -> tf.keras.Model:
    """
    Parameters
    ----------
    input_dim      : number of input features
    n_classes      : output classes
    hidden_layers  : depth of the network
    units          : neurons per hidden layer
    activation     : hidden activation function
    dropout_rate   : fraction of neurons to randomly zero during training
    l1_lambda      : L1 weight decay coefficient
    l2_lambda      : L2 weight decay coefficient
    use_batch_norm : insert BatchNorm before each activation
    use_skip_conn  : add residual skip connections (requires same width)
    """
    reg = _l1l2(l1_lambda, l2_lambda)
    inp = layers.Input(shape=(input_dim,), name="input")
    x   = inp

    for i in range(hidden_layers):
        dense_out = layers.Dense(units, use_bias=not use_batch_norm,
                                 kernel_regularizer=reg,
                                 name=f"dense_{i}")(x)
        if use_batch_norm:
            dense_out = layers.BatchNormalization(name=f"bn_{i}")(dense_out)
        dense_out = layers.Activation(activation, name=f"act_{i}")(dense_out)
        if dropout_rate > 0:
            dense_out = layers.Dropout(dropout_rate, name=f"drop_{i}")(dense_out)

        # Skip connection: only possible when dims match
        if use_skip_conn and x.shape[-1] == units:
            x = layers.Add(name=f"skip_{i}")([x, dense_out])
        else:
            x = dense_out

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return models.Model(inp, out, name="ANN")


# ─────────────────────────────────────────────────────────────────
# CNN  –  Convolutional image classifier  (8×8 grayscale input)
# ─────────────────────────────────────────────────────────────────

def build_cnn(
    *,
    n_classes:      int   = 10,
    n_filters:      int   = 32,
    kernel_size:    int   = 3,
    n_conv_blocks:  int   = 2,
    dense_units:    int   = 128,
    activation:     str   = "relu",
    dropout_rate:   float = 0.0,
    l2_lambda:      float = 0.0,
    use_batch_norm: bool  = False,
    use_global_avg: bool  = False,
) -> tf.keras.Model:
    """
    Parameters
    ----------
    n_filters      : base number of conv filters (doubles each block)
    kernel_size    : convolutional kernel width/height
    n_conv_blocks  : number of Conv→BN→Act→Pool blocks
    dense_units    : units in the fully-connected head
    use_global_avg : replace Flatten with GlobalAvgPool (fewer params)
    """
    reg = _l1l2(0, l2_lambda)
    inp = layers.Input(shape=(8, 8, 1), name="input")
    x   = inp

    for i in range(n_conv_blocks):
        filters = n_filters * (2 ** i)
        x = layers.Conv2D(filters, kernel_size, padding="same",
                          use_bias=not use_batch_norm,
                          kernel_regularizer=reg,
                          name=f"conv_{i}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_conv_{i}")(x)
        x = layers.Activation(activation, name=f"act_conv_{i}")(x)
        # Only pool if spatial dims allow it
        if x.shape[1] > 2:
            x = layers.MaxPooling2D(2, name=f"pool_{i}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"drop_conv_{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x) if use_global_avg \
        else layers.Flatten(name="flatten")(x)

    x = layers.Dense(dense_units, kernel_regularizer=reg,
                     name="fc")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_fc")(x)
    x = layers.Activation(activation, name="act_fc")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="drop_fc")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return models.Model(inp, out, name="CNN")


# ─────────────────────────────────────────────────────────────────
# RNN  –  Sequence-to-one regression
# ─────────────────────────────────────────────────────────────────

def build_rnn(
    *,
    seq_length:       int   = 40,
    rnn_type:         str   = "LSTM",
    rnn_units:        int   = 64,
    n_rnn_layers:     int   = 1,
    dense_units:      int   = 32,
    activation:       str   = "relu",
    dropout_rate:     float = 0.0,
    recurrent_drop:   float = 0.0,
    l2_lambda:        float = 0.0,
    use_batch_norm:   bool  = False,
    bidirectional:    bool  = False,
) -> tf.keras.Model:
    """
    Parameters
    ----------
    rnn_type        : 'LSTM' | 'GRU' | 'SimpleRNN'
    rnn_units       : hidden units per RNN layer
    n_rnn_layers    : stacked RNN depth
    dense_units     : units in the readout MLP
    recurrent_drop  : dropout on recurrent connections (LSTM/GRU only)
    bidirectional   : wrap each RNN layer in Bidirectional()
    """
    reg       = _l1l2(0, l2_lambda)
    rnn_map   = {"LSTM": layers.LSTM, "GRU": layers.GRU,
                 "SimpleRNN": layers.SimpleRNN}
    rnn_cls   = rnn_map.get(rnn_type, layers.LSTM)

    inp = layers.Input(shape=(seq_length, 1), name="input")
    x   = inp

    for i in range(n_rnn_layers):
        return_seq = (i < n_rnn_layers - 1)

        # SimpleRNN doesn't support recurrent_dropout in the same way
        rnn_kwargs = dict(
            units=rnn_units,
            return_sequences=return_seq,
            dropout=dropout_rate,
            kernel_regularizer=reg,
            name=f"{rnn_type.lower()}_{i}",
        )
        if rnn_type in ("LSTM", "GRU"):
            rnn_kwargs["recurrent_dropout"] = recurrent_drop

        rnn_layer = rnn_cls(**rnn_kwargs)

        if bidirectional:
            rnn_layer = layers.Bidirectional(rnn_layer,
                                             name=f"bi_{rnn_type.lower()}_{i}")
        x = rnn_layer(x)

        if use_batch_norm and return_seq:
            x = layers.BatchNormalization(name=f"bn_rnn_{i}")(x)
        if dropout_rate > 0 and return_seq:
            x = layers.Dropout(dropout_rate, name=f"drop_rnn_{i}")(x)

    # Readout MLP
    x   = layers.Dense(dense_units, kernel_regularizer=reg,
                       name="fc")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_fc")(x)
    x   = layers.Activation(activation, name="act_fc")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="drop_fc")(x)
    out = layers.Dense(1, name="output")(x)

    return models.Model(inp, out, name=f"RNN_{rnn_type}")
