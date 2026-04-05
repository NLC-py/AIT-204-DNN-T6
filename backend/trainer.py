"""
backend/trainer.py
Compilation and training logic.
Returns the Keras History object; no Streamlit imports here.
"""

import numpy as np
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────
# Optimizer factory
# ─────────────────────────────────────────────────────────────────

def build_optimizer(
    name:          str,
    learning_rate: float,
    momentum:      float = 0.9,
    nesterov:      bool  = False,
    beta_1:        float = 0.9,
    beta_2:        float = 0.999,
    epsilon:       float = 1e-7,
    rho:           float = 0.9,
    weight_decay:  float = 0.0,
) -> tf.keras.optimizers.Optimizer:
    """
    Supported optimizers
    --------------------
    Adam      – adaptive per-parameter lr, momentum + RMSprop combined
    AdamW     – Adam with decoupled weight decay regularisation
    SGD       – vanilla gradient descent  (+ optional Nesterov momentum)
    RMSprop   – adaptive lr via running mean of squared gradients
    Adagrad   – accumulates squared gradients; good for sparse data
    Adadelta  – extension of Adagrad with windowed gradient accumulation
    """
    lr = learning_rate
    name = name.lower()

    if name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )
    if name == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=max(weight_decay, 1e-6),
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )
    if name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=momentum, nesterov=nesterov
        )
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=lr, rho=rho, epsilon=epsilon, momentum=momentum
        )
    if name == "adagrad":
        return tf.keras.optimizers.Adagrad(
            learning_rate=lr, epsilon=epsilon
        )
    if name == "adadelta":
        return tf.keras.optimizers.Adadelta(
            learning_rate=lr, rho=rho, epsilon=epsilon
        )
    raise ValueError(f"Unknown optimizer: {name}")


# ─────────────────────────────────────────────────────────────────
# Learning-rate schedule factory
# ─────────────────────────────────────────────────────────────────

def build_lr_schedule(
    schedule_name: str,
    base_lr:       float,
    epochs:        int,
) -> tf.keras.callbacks.Callback | None:
    """
    none         – constant LR throughout
    step_decay   – halve LR every 1/3 of training
    cosine       – smooth cosine annealing to near-zero
    reduce_on_plateau – cut LR when val_loss stops improving
    warmup_cosine – linear warm-up for 10 % then cosine decay
    """
    if schedule_name == "none":
        return None

    if schedule_name == "step_decay":
        step = max(1, epochs // 3)
        def schedule(epoch, lr):
            return lr * 0.5 if (epoch > 0 and epoch % step == 0) else lr
        return tf.keras.callbacks.LearningRateScheduler(schedule)

    if schedule_name == "cosine":
        def cosine_schedule(epoch, lr):
            return float(
                base_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
            )
        return tf.keras.callbacks.LearningRateScheduler(cosine_schedule)

    if schedule_name == "reduce_on_plateau":
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, epochs // 10),
            min_lr=base_lr * 1e-3, verbose=0
        )

    if schedule_name == "warmup_cosine":
        warmup_epochs = max(1, int(0.1 * epochs))
        def warmup_cosine_schedule(epoch, lr):
            if epoch < warmup_epochs:
                return float(base_lr * (epoch + 1) / warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return float(base_lr * 0.5 * (1 + np.cos(np.pi * progress)))
        return tf.keras.callbacks.LearningRateScheduler(warmup_cosine_schedule)

    return None


# ─────────────────────────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────────────────────────

def train_model(
    model:           tf.keras.Model,
    X_train,         y_train,
    X_val,           y_val,
    *,
    optimizer_name:  str   = "adam",
    learning_rate:   float = 0.001,
    epochs:          int   = 50,
    batch_size:      int   = 32,
    is_regression:   bool  = False,
    # Optimizer kwargs
    momentum:        float = 0.9,
    nesterov:        bool  = False,
    beta_1:          float = 0.9,
    beta_2:          float = 0.999,
    epsilon:         float = 1e-7,
    rho:             float = 0.9,
    weight_decay:    float = 0.0,
    # LR scheduling
    lr_schedule:     str   = "none",
    # Misc
    early_stopping:  bool  = False,
    patience:        int   = 10,
    use_gradient_clip: bool  = False,
    clip_value:      float = 1.0,
    class_weights:   dict | None = None,
) -> tf.keras.callbacks.History:
    """
    Compiles and trains `model` in-place, returns the history object.
    """
    opt = build_optimizer(
        optimizer_name, learning_rate,
        momentum=momentum, nesterov=nesterov,
        beta_1=beta_1, beta_2=beta_2,
        epsilon=epsilon, rho=rho,
        weight_decay=weight_decay,
    )

    if use_gradient_clip:
        opt.clipvalue = clip_value

    if is_regression:
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    else:
        model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    callbacks = []

    lr_cb = build_lr_schedule(lr_schedule, learning_rate, epochs)
    if lr_cb:
        callbacks.append(lr_cb)

    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=0,
            )
        )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    return history


# ─────────────────────────────────────────────────────────────────
# Post-training evaluation helpers
# ─────────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test_cat):
    """Returns loss, accuracy and per-class confusion matrix info."""
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    preds = model.predict(X_test, verbose=0).argmax(axis=1)
    truth = y_test_cat.argmax(axis=1)
    return {
        "test_loss":     round(float(loss), 4),
        "test_accuracy": round(float(acc),  4),
        "predictions":   preds,
        "ground_truth":  truth,
    }


def evaluate_regressor(model, X_test, y_test):
    """Returns MSE, MAE and prediction array."""
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    preds = model.predict(X_test, verbose=0).flatten()
    residuals = y_test - preds
    return {
        "test_mse":  round(float(loss), 5),
        "test_mae":  round(float(mae),  5),
        "test_rmse": round(float(np.sqrt(loss)), 5),
        "predictions": preds,
        "ground_truth": y_test,
        "residuals": residuals,
    }
