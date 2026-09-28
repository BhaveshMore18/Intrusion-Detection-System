"""
model.py

A small feedforward deep neural network for binary classification
(benign=0 / malicious=1), implemented from scratch with NumPy so the
mechanics (forward pass, backprop, Adam updates) are fully visible and
auditable -- not just a call into a black-box library.

Architecture: configurable hidden layers with ReLU activations and a
sigmoid output layer, trained with mini-batch gradient descent + Adam.
"""

import json

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(x.dtype)


def sigmoid(x):
    x = np.clip(x, -500, 500)  # avoid overflow
    return 1.0 / (1.0 + np.exp(-x))


class DeepNeuralNetwork:
    """Feedforward network: input -> [hidden layers w/ ReLU] -> sigmoid output."""

    def __init__(self, input_dim: int, hidden_layers=(32, 16), seed: int = 42):
        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        rng = np.random.default_rng(seed)

        layer_dims = [input_dim] + self.hidden_layers + [1]
        self.weights = []
        self.biases = []
        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
            # He initialization -- suited to ReLU hidden layers
            limit = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        # Adam optimizer state
        self._m_w = [np.zeros_like(W) for W in self.weights]
        self._v_w = [np.zeros_like(W) for W in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._t = 0

    # ---- forward / backward ----

    def _forward(self, X):
        activations = [X]
        zs = []
        a = X
        n_layers = len(self.weights)
        for i in range(n_layers):
            z = a @ self.weights[i] + self.biases[i]
            zs.append(z)
            if i < n_layers - 1:
                a = relu(z)
            else:
                a = sigmoid(z)
            activations.append(a)
        return activations, zs

    def predict_proba(self, X):
        activations, _ = self._forward(X)
        return activations[-1].ravel()

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def _backward(self, activations, zs, y):
        n_layers = len(self.weights)
        m = y.shape[0]
        y = y.reshape(-1, 1)

        grads_w = [None] * n_layers
        grads_b = [None] * n_layers

        # Output layer: binary cross-entropy + sigmoid -> simple gradient
        dz = activations[-1] - y  # (m, 1)
        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            grads_w[i] = (a_prev.T @ dz) / m
            grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                da_prev = dz @ self.weights[i].T
                dz = da_prev * relu_derivative(zs[i - 1])
        return grads_w, grads_b

    @staticmethod
    def _clip_grad_norm(grads_w, grads_b, max_norm=5.0):
        total_sq = sum(np.sum(g ** 2) for g in grads_w) + sum(np.sum(g ** 2) for g in grads_b)
        total_norm = np.sqrt(total_sq)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            grads_w = [g * scale for g in grads_w]
            grads_b = [g * scale for g in grads_b]
        return grads_w, grads_b

    def _adam_step(self, grads_w, grads_b, lr, beta1=0.9, beta2=0.999, eps=1e-8,
                   weight_decay=1e-4):
        grads_w, grads_b = self._clip_grad_norm(grads_w, grads_b)
        self._t += 1
        for i in range(len(self.weights)):
            self._m_w[i] = beta1 * self._m_w[i] + (1 - beta1) * grads_w[i]
            self._v_w[i] = beta2 * self._v_w[i] + (1 - beta2) * (grads_w[i] ** 2)
            m_hat = self._m_w[i] / (1 - beta1 ** self._t)
            v_hat = self._v_w[i] / (1 - beta2 ** self._t)
            self.weights[i] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * self.weights[i])

            self._m_b[i] = beta1 * self._m_b[i] + (1 - beta1) * grads_b[i]
            self._v_b[i] = beta2 * self._v_b[i] + (1 - beta2) * (grads_b[i] ** 2)
            m_hat_b = self._m_b[i] / (1 - beta1 ** self._t)
            v_hat_b = self._v_b[i] / (1 - beta2 ** self._t)
            self.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    @staticmethod
    def _bce_loss(y_pred, y_true):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = y_true.ravel()
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # ---- training loop ----

    def fit(self, X, y, X_val=None, y_val=None, epochs=60, batch_size=64,
            lr=0.01, patience=8, verbose=True):
        n = X.shape[0]
        history = {"loss": [], "val_loss": []}
        rng = np.random.default_rng(0)

        best_val = np.inf
        best_state = None
        epochs_since_best = 0

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            X_shuf, y_shuf = X[perm], y[perm]

            for start in range(0, n, batch_size):
                end = start + batch_size
                Xb, yb = X_shuf[start:end], y_shuf[start:end]

                activations, zs = self._forward(Xb)
                grads_w, grads_b = self._backward(activations, zs, yb)
                self._adam_step(grads_w, grads_b, lr)

            # Report loss on the FULL training set using this epoch's final
            # weights (not the mean of stale per-batch losses seen during
            # the epoch, which is noisier and less interpretable).
            train_pred = self.predict_proba(X)
            train_loss = float(self._bce_loss(train_pred, y))
            history["loss"].append(train_loss)

            val_loss = None
            if X_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self._bce_loss(val_pred, y_val)
                history["val_loss"].append(float(val_loss))

                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = (
                        [W.copy() for W in self.weights],
                        [b.copy() for b in self.biases],
                    )
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1

            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                msg = f"epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}"
                print(msg)

            if X_val is not None and epochs_since_best >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
                break

        if best_state is not None:
            self.weights, self.biases = best_state

        return history

    # ---- persistence ----

    def save(self, path: str):
        payload = {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
        }
        arrays = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(path, meta=json.dumps(payload), **arrays)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        model = cls(meta["input_dim"], tuple(meta["hidden_layers"]))
        n_layers = len(model.weights)
        for i in range(n_layers):
            model.weights[i] = data[f"W{i}"]
            model.biases[i] = data[f"b{i}"]
        return model
