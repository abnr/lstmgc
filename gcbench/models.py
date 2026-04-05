from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except Exception:  # pragma: no cover - exercised indirectly when torch is absent
    torch = None
    nn = None
    HAS_TORCH = False


MODEL_ALIASES = {
    "var": "var",
    "var-gc": "var",
    "regularized_var": "ridge_var",
    "regularized var-gc": "ridge_var",
    "regularized var": "ridge_var",
    "ridge_var": "ridge_var",
    "simple_lstm": "simple_lstm",
    "simple-lstm": "simple_lstm",
    "conv_lstm": "conv_lstm",
    "conv-lstm": "conv_lstm",
    "rnn_gc": "rnn_gc",
    "rnn-gc": "rnn_gc",
    "neural_gc": "neural_gc",
    "neural-gc": "neural_gc",
}


TORCH_REQUIRED_MESSAGE = "Requested neural model requires PyTorch. Use the local venv or install torch first."


def canonical_model_name(name: str) -> str:
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        valid = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model '{name}'. Expected one of: {valid}")
    return MODEL_ALIASES[key]


def implemented_model_names() -> tuple[str, ...]:
    base = ["var", "ridge_var"]
    if HAS_TORCH:
        base.extend(["simple_lstm", "conv_lstm", "rnn_gc", "neural_gc"])
    return tuple(base)


def known_model_names() -> tuple[str, ...]:
    return tuple(sorted(set(MODEL_ALIASES.values())))


@dataclass
class LinearAutoregressiveRegressor:
    max_lag: int
    ridge: float = 0.0
    target_: int | None = None
    included_sources_: tuple[int, ...] | None = None
    feature_mean_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None
    weights_: np.ndarray | None = None
    bias_: float = 0.0

    def fit(
        self,
        data: np.ndarray,
        target: int,
        *,
        included_sources: tuple[int, ...] | None = None,
        val_data: np.ndarray | None = None,
    ) -> "LinearAutoregressiveRegressor":
        del val_data
        if data.ndim != 2:
            raise ValueError("Expected data with shape (time, variables).")
        if data.shape[0] <= self.max_lag:
            raise ValueError("Not enough time points to fit the requested lag order.")

        n_vars = data.shape[1]
        if included_sources is None:
            included_sources = tuple(range(n_vars))

        features = build_lagged_features(data, self.max_lag, included_sources)
        target_values = build_target_vector(data, target, self.max_lag)

        feature_mean = features.mean(axis=0, keepdims=True)
        feature_scale = features.std(axis=0, keepdims=True)
        feature_scale[feature_scale == 0.0] = 1.0
        normalized = (features - feature_mean) / feature_scale

        ones = np.ones((normalized.shape[0], 1), dtype=float)
        design = np.concatenate([normalized, ones], axis=1)
        penalty = np.eye(design.shape[1], dtype=float) * self.ridge
        penalty[-1, -1] = 0.0

        lhs = design.T @ design + penalty
        rhs = design.T @ target_values
        params = np.linalg.solve(lhs, rhs)

        self.target_ = target
        self.included_sources_ = tuple(included_sources)
        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        self.weights_ = params[:-1]
        self.bias_ = float(params[-1])
        return self

    def predict_segment(self, history: np.ndarray, segment: np.ndarray) -> np.ndarray:
        self._check_fitted()
        history = np.asarray(history, dtype=float)
        segment = np.asarray(segment, dtype=float)
        if segment.ndim != 2:
            raise ValueError("segment must have shape (time, variables)")
        if history.ndim != 2:
            raise ValueError("history must have shape (time, variables)")
        if history.shape[1] != segment.shape[1]:
            raise ValueError("history and segment must have the same number of variables")

        context = history[-self.max_lag :] if history.shape[0] >= self.max_lag else history
        combined = np.concatenate([context, segment], axis=0)
        if combined.shape[0] <= self.max_lag:
            raise ValueError("Not enough context to build lagged predictions.")

        features = build_lagged_features(combined, self.max_lag, self.included_sources_)
        normalized = (features - self.feature_mean_) / self.feature_scale_
        return normalized @ self.weights_ + self.bias_

    def mse_segment(self, history: np.ndarray, segment: np.ndarray) -> float:
        predictions = self.predict_segment(history, segment)
        target = segment[:, self.target_]
        return float(np.mean((predictions - target) ** 2))

    def _check_fitted(self) -> None:
        if self.weights_ is None or self.target_ is None or self.included_sources_ is None:
            raise RuntimeError("Model is not fitted yet.")


@dataclass
class TorchSequenceRegressor:
    max_lag: int
    architecture: str
    hidden_size: int = 15
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 8
    weight_decay: float = 0.0
    random_state: int | None = None
    conv_channels: tuple[int, int, int] = (64, 128, 256)
    mlp_hidden: int = 64
    target_: int | None = None
    included_sources_: tuple[int, ...] | None = None
    feature_mean_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None
    target_mean_: float = 0.0
    target_scale_: float = 1.0
    module_: nn.Module | None = None

    def fit(
        self,
        data: np.ndarray,
        target: int,
        *,
        included_sources: tuple[int, ...] | None = None,
        val_data: np.ndarray | None = None,
    ) -> "TorchSequenceRegressor":
        self._require_torch()
        if data.ndim != 2:
            raise ValueError("Expected data with shape (time, variables).")
        if data.shape[0] <= self.max_lag:
            raise ValueError("Not enough time points to fit the requested lag order.")

        n_vars = data.shape[1]
        if included_sources is None:
            included_sources = tuple(range(n_vars))

        x_train, y_train = build_sequence_dataset(data, self.max_lag, included_sources, target)
        x_val, y_val = self._build_validation_data(data, val_data, included_sources, target)

        self.feature_mean_ = x_train.mean(axis=(0, 1), keepdims=True)
        self.feature_scale_ = x_train.std(axis=(0, 1), keepdims=True)
        self.feature_scale_[self.feature_scale_ == 0.0] = 1.0
        self.target_mean_ = float(y_train.mean())
        self.target_scale_ = float(y_train.std()) or 1.0

        x_train = ((x_train - self.feature_mean_) / self.feature_scale_).astype(np.float32)
        x_val = ((x_val - self.feature_mean_) / self.feature_scale_).astype(np.float32)
        y_train = ((y_train - self.target_mean_) / self.target_scale_).astype(np.float32)
        y_val = ((y_val - self.target_mean_) / self.target_scale_).astype(np.float32)

        self.target_ = target
        self.included_sources_ = tuple(included_sources)
        self.module_ = self._build_module(input_dim=len(included_sources))
        self._train_module(x_train, y_train, x_val, y_val)
        return self

    def predict_segment(self, history: np.ndarray, segment: np.ndarray) -> np.ndarray:
        self._check_fitted()
        history = np.asarray(history, dtype=float)
        segment = np.asarray(segment, dtype=float)
        context = history[-self.max_lag :] if history.shape[0] >= self.max_lag else history
        combined = np.concatenate([context, segment], axis=0)
        x_pred, _ = build_sequence_dataset(combined, self.max_lag, self.included_sources_, self.target_)
        x_pred = ((x_pred - self.feature_mean_) / self.feature_scale_).astype(np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(x_pred)
            predictions = self.module_(tensor).cpu().numpy().reshape(-1)
        return predictions * self.target_scale_ + self.target_mean_

    def mse_segment(self, history: np.ndarray, segment: np.ndarray) -> float:
        predictions = self.predict_segment(history, segment)
        target = segment[:, self.target_]
        return float(np.mean((predictions - target) ** 2))

    def _build_validation_data(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None,
        included_sources: tuple[int, ...],
        target: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if val_data is not None and val_data.shape[0] > 0:
            context = train_data[-self.max_lag :]
            combined = np.concatenate([context, val_data], axis=0)
            x_val, y_val = build_sequence_dataset(combined, self.max_lag, included_sources, target)
            if x_val.shape[0] > 0:
                return x_val, y_val

        x_train, y_train = build_sequence_dataset(train_data, self.max_lag, included_sources, target)
        split = max(1, int(0.8 * x_train.shape[0]))
        if split >= x_train.shape[0]:
            return x_train, y_train
        return x_train[split:], y_train[split:]

    def _build_module(self, input_dim: int) -> nn.Module:
        if self.architecture == "simple_lstm":
            return _SimpleLSTMNet(input_dim=input_dim, hidden_size=self.hidden_size)
        if self.architecture == "conv_lstm":
            return _ConvLSTMNet(input_dim=input_dim, hidden_size=self.hidden_size, conv_channels=self.conv_channels)
        if self.architecture == "rnn_gc":
            return _SimpleRNNNet(input_dim=input_dim, hidden_size=self.hidden_size)
        if self.architecture == "neural_gc":
            return _FeedForwardGCNet(input_dim=input_dim, max_lag=self.max_lag, hidden_size=self.mlp_hidden)
        raise ValueError(f"Unknown torch architecture: {self.architecture}")

    def _train_module(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        x_train_tensor = torch.from_numpy(x_train)
        y_train_tensor = torch.from_numpy(y_train.reshape(-1, 1))
        x_val_tensor = torch.from_numpy(x_val)
        y_val_tensor = torch.from_numpy(y_val.reshape(-1, 1))

        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best_state = {name: tensor.detach().clone() for name, tensor in self.module_.state_dict().items()}
        best_val = float("inf")
        patience_left = self.patience

        for _epoch in range(self.epochs):
            self.module_.train()
            for batch_idx in _iter_minibatches(x_train_tensor.shape[0], self.batch_size):
                xb = x_train_tensor[batch_idx]
                yb = y_train_tensor[batch_idx]
                optimizer.zero_grad()
                predictions = self.module_(xb)
                loss = loss_fn(predictions, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), max_norm=1.0)
                optimizer.step()

            self.module_.eval()
            with torch.no_grad():
                val_predictions = self.module_(x_val_tensor)
                val_loss = float(loss_fn(val_predictions, y_val_tensor).item())

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {name: tensor.detach().clone() for name, tensor in self.module_.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.module_.load_state_dict(best_state)
        self.module_.eval()

    def _check_fitted(self) -> None:
        if self.module_ is None or self.included_sources_ is None or self.target_ is None:
            raise RuntimeError("Model is not fitted yet.")

    @staticmethod
    def _require_torch() -> None:
        if not HAS_TORCH:
            raise RuntimeError(TORCH_REQUIRED_MESSAGE)


def build_lagged_features(data: np.ndarray, max_lag: int, included_sources: tuple[int, ...]) -> np.ndarray:
    rows = data.shape[0] - max_lag
    cols = len(included_sources) * max_lag
    features = np.empty((rows, cols), dtype=float)
    column = 0

    for source in included_sources:
        for lag in range(1, max_lag + 1):
            features[:, column] = data[max_lag - lag : data.shape[0] - lag, source]
            column += 1

    return features


def build_target_vector(data: np.ndarray, target: int, max_lag: int) -> np.ndarray:
    return np.asarray(data[max_lag:, target], dtype=float)


def build_sequence_dataset(
    data: np.ndarray,
    max_lag: int,
    included_sources: tuple[int, ...],
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    if data.shape[0] <= max_lag:
        raise ValueError("Not enough time points to build sequence windows.")
    rows = data.shape[0] - max_lag
    x = np.empty((rows, max_lag, len(included_sources)), dtype=float)
    for row in range(rows):
        x[row] = data[row : row + max_lag, included_sources]
    y = data[max_lag:, target].astype(float)
    return x, y


def _iter_minibatches(n_rows: int, batch_size: int) -> list[np.ndarray]:
    if n_rows <= batch_size:
        return [np.arange(n_rows)]
    order = np.random.permutation(n_rows)
    return [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]


if HAS_TORCH:

    class _SimpleLSTMNet(nn.Module):
        def __init__(self, *, input_dim: int, hidden_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (hidden, _) = self.lstm(x)
            return self.output(hidden[-1])


    class _ConvLSTMNet(nn.Module):
        def __init__(self, *, input_dim: int, hidden_size: int, conv_channels: tuple[int, int, int]) -> None:
            super().__init__()
            c1, c2, c3 = conv_channels
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=c1, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=c2, out_channels=c3, kernel_size=1),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(input_size=c3, hidden_size=hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)
            _, (hidden, _) = self.lstm(x)
            return self.output(hidden[-1])


    class _SimpleRNNNet(nn.Module):
        def __init__(self, *, input_dim: int, hidden_size: int) -> None:
            super().__init__()
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_size, nonlinearity="tanh", batch_first=True)
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, hidden = self.rnn(x)
            return self.output(hidden[-1])


    class _FeedForwardGCNet(nn.Module):
        def __init__(self, *, input_dim: int, max_lag: int, hidden_size: int) -> None:
            super().__init__()
            flattened = input_dim * max_lag
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def create_model(
    name: str,
    max_lag: int,
    *,
    random_state: int | None = None,
    **model_kwargs: object,
) -> LinearAutoregressiveRegressor | TorchSequenceRegressor:
    canonical = canonical_model_name(name)
    if canonical == "var":
        return LinearAutoregressiveRegressor(max_lag=max_lag, ridge=0.0)
    if canonical == "ridge_var":
        return LinearAutoregressiveRegressor(max_lag=max_lag, ridge=1e-2)
    if canonical in {"simple_lstm", "conv_lstm", "rnn_gc", "neural_gc"}:
        if not HAS_TORCH:
            raise RuntimeError(TORCH_REQUIRED_MESSAGE)
        return TorchSequenceRegressor(
            max_lag=max_lag,
            architecture=canonical,
            random_state=random_state,
            **model_kwargs,
        )
    raise RuntimeError(f"Unsupported model configuration: {name}")
