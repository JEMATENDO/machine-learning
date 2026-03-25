import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CausalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.padding > 0:
            y = y[:, :, :-self.padding]
        y = self.relu(y)
        y = self.dropout(y)

        residual = x if self.residual is None else self.residual(x)
        return y + residual


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for d in dilations:
            blocks.append(CausalResidualBlock(in_channels, hidden_dim, kernel_size, d, dropout))
            in_channels = hidden_dim
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)  # (batch, seq_len, hidden)


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int = 64, head_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = z.mean(dim=1)
        return self.net(pooled).squeeze(-1)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int = 64, head_dim: int = 32, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = z.mean(dim=1)
        return self.net(pooled)


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


@dataclass
class ExperimentConfig:
    name: str
    drop_features: Tuple[str, ...] = ()
    mode: str = "multitask"  # multitask | classification | regression
    alpha: float = 0.7
    beta: float = 0.3
    weighted_ce: bool = False
    focal_gamma: float = 0.0
    use_weighted_sampler: bool = False
    freeze_encoder: bool = False
    unfreeze_encoder_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    max_epochs: int = 30
    patience: int = 6


def _cfg_from_dict(data: Dict) -> ExperimentConfig:
    return ExperimentConfig(
        name=data["name"],
        drop_features=tuple(data.get("drop_features", [])),
        mode=data.get("mode", "multitask"),
        alpha=float(data.get("alpha", 0.7)),
        beta=float(data.get("beta", 0.3)),
        weighted_ce=bool(data.get("weighted_ce", False)),
        focal_gamma=float(data.get("focal_gamma", 0.0)),
        use_weighted_sampler=bool(data.get("use_weighted_sampler", False)),
        freeze_encoder=bool(data.get("freeze_encoder", False)),
        unfreeze_encoder_lr=float(data.get("unfreeze_encoder_lr", 1e-4)),
        head_lr=float(data.get("head_lr", 1e-3)),
        weight_decay=float(data.get("weight_decay", 1e-4)),
        dropout=float(data.get("dropout", 0.2)),
        max_epochs=int(data.get("max_epochs", 30)),
        patience=int(data.get("patience", 6)),
    )


@dataclass
class ExperimentResult:
    config: Dict
    val_objective: float
    test_metrics: Dict


class Runner:
    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device

        self.X_train = np.load(root / "X_train_windows.npy")
        self.X_val = np.load(root / "X_val_windows.npy")
        self.X_test = np.load(root / "X_test_windows.npy")

        self.feature_names = self._load_feature_names()

        self.y_train_vol = np.load(root / "y_train_vol_windows.npy").reshape(-1)
        self.y_val_vol = np.load(root / "y_val_vol_windows.npy").reshape(-1)
        self.y_test_vol = np.load(root / "y_test_vol_windows.npy").reshape(-1)

        self.y_train_cong = np.load(root / "y_train_cong_windows.npy").reshape(-1).astype(np.int64)
        self.y_val_cong = np.load(root / "y_val_cong_windows.npy").reshape(-1).astype(np.int64)
        self.y_test_cong = np.load(root / "y_test_cong_windows.npy").reshape(-1).astype(np.int64)

        self.num_classes = int(max(self.y_train_cong.max(), self.y_val_cong.max(), self.y_test_cong.max()) + 1)

    def _load_feature_names(self) -> List[str]:
        meta_path = self.root / "data_package.json"
        if not meta_path.exists():
            return [f"f{i}" for i in range(self.X_train.shape[-1])]
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("tcn_input_features", [f"f{i}" for i in range(self.X_train.shape[-1])])

    def _selected_columns(self, drop_features: Tuple[str, ...]) -> np.ndarray:
        drop_set = set(drop_features)
        keep = [i for i, name in enumerate(self.feature_names) if name not in drop_set]
        if not keep:
            raise ValueError("All features were dropped. Please keep at least one input feature.")
        return np.array(keep, dtype=np.int64)

    def _make_loaders(
        self, columns: np.ndarray, cfg: ExperimentConfig, batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        def to_dataset(X: np.ndarray, yv: np.ndarray, yc: np.ndarray) -> TensorDataset:
            return TensorDataset(
                torch.tensor(X[:, :, columns], dtype=torch.float32),
                torch.tensor(yv, dtype=torch.float32),
                torch.tensor(yc, dtype=torch.long),
            )

        train_ds = to_dataset(self.X_train, self.y_train_vol, self.y_train_cong)
        val_ds = to_dataset(self.X_val, self.y_val_vol, self.y_val_cong)
        test_ds = to_dataset(self.X_test, self.y_test_vol, self.y_test_cong)

        if cfg.use_weighted_sampler and cfg.mode in {"multitask", "classification"}:
            class_counts = np.bincount(self.y_train_cong, minlength=self.num_classes).astype(np.float64)
            class_counts[class_counts == 0] = 1.0
            class_weights = class_counts.sum() / class_counts
            sample_weights = class_weights[self.y_train_cong]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def _class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.y_train_cong, minlength=self.num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def diagnostics(self) -> Dict:
        def split_dist(y: np.ndarray) -> List[Dict]:
            counts = np.bincount(y, minlength=self.num_classes)
            total = int(counts.sum()) if counts.sum() > 0 else 1
            return [
                {
                    "class": int(c),
                    "count": int(counts[c]),
                    "pct": float(counts[c] / total),
                }
                for c in range(self.num_classes)
            ]

        output = {
            "class_distribution": {
                "train": split_dist(self.y_train_cong),
                "val": split_dist(self.y_val_cong),
                "test": split_dist(self.y_test_cong),
            }
        }

        if "vehicle_count" in self.feature_names:
            vc_idx = self.feature_names.index("vehicle_count")
            x_last = self.X_train[:, -1, vc_idx]
            corr = np.corrcoef(x_last, self.y_train_vol)[0, 1]
            output["leakage_signal"] = {
                "feature": "vehicle_count",
                "corr_last_step_with_target_volume": float(corr),
            }

        return output

    def _load_phase1_encoder(self, encoder: TCNEncoder) -> bool:
        ckpt_path = self.root / "phase1_checkpoint.pt"
        if not ckpt_path.exists():
            return False

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)

            normalized = {}
            for k, v in state_dict.items():
                key = k
                if key.startswith("encoder."):
                    key = key[len("encoder.") :]
                normalized[key] = v

            # Keep only keys with matching tensor shapes so feature-drop experiments
            # can still reuse most pretrained layers.
            current_state = encoder.state_dict()
            compatible = {
                k: v
                for k, v in normalized.items()
                if (k in current_state and current_state[k].shape == v.shape)
            }

            missing, unexpected = encoder.load_state_dict(compatible, strict=False)
            if len(unexpected) > 0:
                print(f"[WARN] Unexpected checkpoint keys ignored: {len(unexpected)}")
            if len(missing) > 0:
                print(f"[WARN] Missing checkpoint keys: {len(missing)}")
            return len(compatible) > 0
        except Exception as exc:
            print(f"[WARN] Failed to load phase1 checkpoint: {exc}")
            return False

    @staticmethod
    def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"mae": mae, "rmse": rmse}

    def _compute_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        pr, rc, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        per_class = []
        for c in range(self.num_classes):
            per_class.append(
                {
                    "class": int(c),
                    "precision": float(pr[c]),
                    "recall": float(rc[c]),
                    "f1": float(f1[c]),
                    "support": int(support[c]),
                }
            )

        target_class = 4 if self.num_classes > 4 else (self.num_classes - 1)

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": bal_acc,
            "target_class": int(target_class),
            "target_class_recall": float(rc[target_class]),
            "target_class_f1": float(f1[target_class]),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }

    def _train_core(self, cfg: ExperimentConfig, columns: np.ndarray):
        train_loader, val_loader, test_loader = self._make_loaders(columns, cfg=cfg)

        input_dim = len(columns)
        encoder = TCNEncoder(input_dim=input_dim, dropout=cfg.dropout).to(self.device)
        regression_head = RegressionHead().to(self.device)
        classification_head = ClassificationHead(num_classes=self.num_classes).to(self.device)

        loaded_pretrained = self._load_phase1_encoder(encoder)

        if cfg.freeze_encoder and loaded_pretrained:
            for p in encoder.parameters():
                p.requires_grad = False
        elif cfg.freeze_encoder and not loaded_pretrained:
            print(f"[WARN] {cfg.name}: freeze_encoder requested but no pretrained checkpoint loaded; encoder left trainable.")

        criterion_reg = nn.L1Loss()
        class_weights = self._class_weights() if cfg.weighted_ce else None
        if cfg.focal_gamma > 0:
            criterion_clf = FocalCrossEntropyLoss(gamma=cfg.focal_gamma, weight=class_weights)
        elif cfg.weighted_ce:
            criterion_clf = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion_clf = nn.CrossEntropyLoss()

        params = []
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        if enc_params:
            params.append({"params": enc_params, "lr": cfg.unfreeze_encoder_lr})

        if cfg.mode in {"multitask", "regression"}:
            params.append({"params": regression_head.parameters(), "lr": cfg.head_lr})
        if cfg.mode in {"multitask", "classification"}:
            params.append({"params": classification_head.parameters(), "lr": cfg.head_lr})

        optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

        best_val = math.inf
        best_snapshot = None
        no_improve = 0

        for _epoch in range(cfg.max_epochs):
            encoder.train()
            regression_head.train()
            classification_head.train()

            for xb, yv, yc in train_loader:
                xb = xb.to(self.device)
                yv = yv.to(self.device)
                yc = yc.to(self.device)

                z = encoder(xb)

                losses = []
                if cfg.mode in {"multitask", "regression"}:
                    vol_pred = regression_head(z)
                    loss_reg = criterion_reg(vol_pred, yv)
                    losses.append(cfg.alpha * loss_reg if cfg.mode == "multitask" else loss_reg)

                if cfg.mode in {"multitask", "classification"}:
                    cong_logits = classification_head(z)
                    loss_clf = criterion_clf(cong_logits, yc)
                    losses.append(cfg.beta * loss_clf if cfg.mode == "multitask" else loss_clf)

                loss = sum(losses)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_objective = self._validate(cfg, encoder, regression_head, classification_head, val_loader, criterion_reg, criterion_clf)
            if val_objective < best_val:
                best_val = val_objective
                no_improve = 0
                best_snapshot = {
                    "encoder": encoder.state_dict(),
                    "reg_head": regression_head.state_dict(),
                    "clf_head": classification_head.state_dict(),
                }
            else:
                no_improve += 1

            if no_improve >= cfg.patience:
                break

        if best_snapshot is not None:
            encoder.load_state_dict(best_snapshot["encoder"])
            regression_head.load_state_dict(best_snapshot["reg_head"])
            classification_head.load_state_dict(best_snapshot["clf_head"])

        test_metrics = self._evaluate(cfg, encoder, regression_head, classification_head, test_loader)
        result = ExperimentResult(config=asdict(cfg), val_objective=float(best_val), test_metrics=test_metrics)

        return result, {
            "columns": columns.tolist(),
            "encoder": encoder,
            "regression_head": regression_head,
            "classification_head": classification_head,
        }

    def run_experiment(self, cfg: ExperimentConfig) -> ExperimentResult:
        columns = self._selected_columns(cfg.drop_features)
        result, _artifacts = self._train_core(cfg, columns)
        return result

    def fit_and_export(self, cfg: ExperimentConfig, export_file: Path, role: str) -> ExperimentResult:
        columns = self._selected_columns(cfg.drop_features)
        result, artifacts = self._train_core(cfg, columns)

        payload = {
            "role": role,
            "config": asdict(cfg),
            "columns": artifacts["columns"],
            "feature_names": self.feature_names,
            "num_classes": self.num_classes,
            "encoder_state": artifacts["encoder"].state_dict(),
            "regression_head_state": artifacts["regression_head"].state_dict(),
            "classification_head_state": artifacts["classification_head"].state_dict(),
            "metrics": result.test_metrics,
        }
        torch.save(payload, export_file)
        return result

    def _validate(
        self,
        cfg: ExperimentConfig,
        encoder: TCNEncoder,
        reg_head: RegressionHead,
        clf_head: ClassificationHead,
        loader: DataLoader,
        criterion_reg: nn.Module,
        criterion_clf: nn.Module,
    ) -> float:
        encoder.eval()
        reg_head.eval()
        clf_head.eval()

        reg_losses = []
        clf_losses = []
        with torch.no_grad():
            for xb, yv, yc in loader:
                xb = xb.to(self.device)
                yv = yv.to(self.device)
                yc = yc.to(self.device)
                z = encoder(xb)

                if cfg.mode in {"multitask", "regression"}:
                    vol_pred = reg_head(z)
                    reg_losses.append(float(criterion_reg(vol_pred, yv).item()))
                if cfg.mode in {"multitask", "classification"}:
                    cong_logits = clf_head(z)
                    clf_losses.append(float(criterion_clf(cong_logits, yc).item()))

        reg_loss = float(np.mean(reg_losses)) if reg_losses else 0.0
        clf_loss = float(np.mean(clf_losses)) if clf_losses else 0.0

        if cfg.mode == "regression":
            return reg_loss
        if cfg.mode == "classification":
            return clf_loss
        return cfg.alpha * reg_loss + cfg.beta * clf_loss

    def _evaluate(
        self,
        cfg: ExperimentConfig,
        encoder: TCNEncoder,
        reg_head: RegressionHead,
        clf_head: ClassificationHead,
        loader: DataLoader,
    ) -> Dict:
        encoder.eval()
        reg_head.eval()
        clf_head.eval()

        y_vol_true: List[float] = []
        y_vol_pred: List[float] = []
        y_cong_true: List[int] = []
        y_cong_pred: List[int] = []

        with torch.no_grad():
            for xb, yv, yc in loader:
                xb = xb.to(self.device)
                z = encoder(xb)

                if cfg.mode in {"multitask", "regression"}:
                    vol_pred = reg_head(z).cpu().numpy()
                    y_vol_pred.extend(vol_pred.tolist())
                    y_vol_true.extend(yv.numpy().tolist())

                if cfg.mode in {"multitask", "classification"}:
                    logits = clf_head(z)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    y_cong_pred.extend(pred.tolist())
                    y_cong_true.extend(yc.numpy().tolist())

        output = {}
        if y_vol_true:
            output.update(self._compute_regression_metrics(np.array(y_vol_true), np.array(y_vol_pred)))
        if y_cong_true:
            output.update(self._compute_classification_metrics(np.array(y_cong_true), np.array(y_cong_pred)))
        return output


def default_experiments() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(name="exp01_baseline_multitask", mode="multitask", alpha=0.7, beta=0.3, drop_features=()),
        ExperimentConfig(
            name="exp02_remove_vehicle_count",
            mode="multitask",
            alpha=0.7,
            beta=0.3,
            drop_features=("vehicle_count",),
        ),
        ExperimentConfig(
            name="exp03_remove_vehicle_count_weighted_ce",
            mode="multitask",
            alpha=0.5,
            beta=0.5,
            weighted_ce=True,
            drop_features=("vehicle_count",),
        ),
        ExperimentConfig(
            name="exp04_classification_only_weighted",
            mode="classification",
            weighted_ce=True,
            drop_features=("vehicle_count",),
        ),
        ExperimentConfig(
            name="exp05_regression_only",
            mode="regression",
            drop_features=("vehicle_count",),
        ),
        ExperimentConfig(
            name="exp06_classification_focal_sampler",
            mode="classification",
            weighted_ce=True,
            focal_gamma=2.0,
            use_weighted_sampler=True,
            drop_features=("vehicle_count",),
            max_epochs=20,
            patience=5,
        ),
        ExperimentConfig(
            name="exp07_multitask_focal_sampler",
            mode="multitask",
            alpha=0.6,
            beta=0.4,
            weighted_ce=True,
            focal_gamma=1.5,
            use_weighted_sampler=True,
            drop_features=("vehicle_count",),
            max_epochs=20,
            patience=5,
        ),
    ]


def classification_sweep_experiments() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="sweep_cls_g1p0_lr1e3_do0p2",
            mode="classification",
            weighted_ce=True,
            focal_gamma=1.0,
            use_weighted_sampler=True,
            drop_features=("vehicle_count",),
            head_lr=1e-3,
            dropout=0.2,
            max_epochs=18,
            patience=4,
        ),
        ExperimentConfig(
            name="sweep_cls_g1p5_lr8e4_do0p3",
            mode="classification",
            weighted_ce=True,
            focal_gamma=1.5,
            use_weighted_sampler=True,
            drop_features=("vehicle_count",),
            head_lr=8e-4,
            dropout=0.3,
            max_epochs=18,
            patience=4,
        ),
        ExperimentConfig(
            name="sweep_cls_g2p0_lr6e4_do0p35",
            mode="classification",
            weighted_ce=True,
            focal_gamma=2.0,
            use_weighted_sampler=True,
            drop_features=("vehicle_count",),
            head_lr=6e-4,
            dropout=0.35,
            max_epochs=18,
            patience=4,
        ),
        ExperimentConfig(
            name="sweep_cls_g2p5_lr1e3_do0p35",
            mode="classification",
            weighted_ce=True,
            focal_gamma=2.5,
            use_weighted_sampler=True,
            drop_features=("vehicle_count",),
            head_lr=1e-3,
            dropout=0.35,
            max_epochs=18,
            patience=4,
        ),
    ]


def _select_best_regression(results: List[ExperimentResult]) -> Optional[ExperimentResult]:
    candidates = [r for r in results if r.test_metrics.get("mae") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (r.test_metrics.get("mae", math.inf), r.val_objective))


def _select_best_classification(results: List[ExperimentResult]) -> Optional[ExperimentResult]:
    candidates = [r for r in results if r.test_metrics.get("accuracy") is not None]
    if not candidates:
        return None
    # Prefer hard-congestion capture first, then macro quality, then objective.
    return max(
        candidates,
        key=lambda r: (
            r.test_metrics.get("target_class_recall", -1.0),
            r.test_metrics.get("macro_f1", -1.0),
            r.test_metrics.get("accuracy", -1.0),
            -r.val_objective,
        ),
    )


def summarize(results: List[ExperimentResult]) -> List[Dict]:
    summary = []
    for r in results:
        row = {
            "name": r.config["name"],
            "val_objective": r.val_objective,
            "mae": r.test_metrics.get("mae"),
            "rmse": r.test_metrics.get("rmse"),
            "accuracy": r.test_metrics.get("accuracy"),
            "macro_f1": r.test_metrics.get("macro_f1"),
            "balanced_accuracy": r.test_metrics.get("balanced_accuracy"),
            "target_class_recall": r.test_metrics.get("target_class_recall"),
        }
        summary.append(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick recovery experiments for the TCN project")
    parser.add_argument("--root", type=str, default=".", help="Project root containing .npy arrays")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="quick_recovery_results.json")
    parser.add_argument("--run-classification-sweep", action="store_true")
    parser.add_argument("--export-dual-models", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(args.root).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    runner = Runner(root=root, device=device)
    diagnostics = runner.diagnostics()
    leakage = diagnostics.get("leakage_signal", {})
    if leakage:
        print(
            "Leakage diagnostic: "
            f"corr(last_step_{leakage['feature']}, target_volume)="
            f"{leakage['corr_last_step_with_target_volume']:.4f}"
        )
    experiments = default_experiments()
    if args.run_classification_sweep:
        experiments.extend(classification_sweep_experiments())

    all_results: List[ExperimentResult] = []
    for cfg in experiments:
        print(f"\nRunning {cfg.name} ...")
        result = runner.run_experiment(cfg)
        all_results.append(result)

        acc = result.test_metrics.get("accuracy")
        macro_f1 = result.test_metrics.get("macro_f1")
        mae = result.test_metrics.get("mae")
        rmse = result.test_metrics.get("rmse")
        print(
            f"  val={result.val_objective:.4f} "
            f"mae={mae if mae is not None else 'NA'} rmse={rmse if rmse is not None else 'NA'} "
            f"acc={acc if acc is not None else 'NA'} macro_f1={macro_f1 if macro_f1 is not None else 'NA'} "
            f"class4_recall={result.test_metrics.get('target_class_recall', 'NA')}"
        )

    payload = {
        "seed": args.seed,
        "device": str(device),
        "feature_names": runner.feature_names,
        "diagnostics": diagnostics,
        "experiments": [
            {
                "config": r.config,
                "val_objective": r.val_objective,
                "test_metrics": r.test_metrics,
            }
            for r in all_results
        ],
        "summary": summarize(all_results),
    }

    out_path = root / args.output
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to: {out_path}")

    if args.export_dual_models:
        best_reg = _select_best_regression(all_results)
        best_cls = _select_best_classification(all_results)

        dual_dir = root / "quick_recovery_artifacts"
        dual_dir.mkdir(parents=True, exist_ok=True)

        dual_summary: Dict = {
            "seed": args.seed,
            "results_file": str(out_path),
            "regression_model_file": None,
            "classification_model_file": None,
            "regression_choice": None,
            "classification_choice": None,
        }

        if best_reg is not None:
            reg_cfg = _cfg_from_dict(best_reg.config)
            reg_path = dual_dir / "regression_model.pt"
            reg_result = runner.fit_and_export(reg_cfg, reg_path, role="regression")
            dual_summary["regression_model_file"] = str(reg_path)
            dual_summary["regression_choice"] = {
                "config": reg_result.config,
                "metrics": reg_result.test_metrics,
            }
            print(f"Exported regression model to: {reg_path}")

        if best_cls is not None:
            cls_cfg = _cfg_from_dict(best_cls.config)
            cls_path = dual_dir / "classification_model.pt"
            cls_result = runner.fit_and_export(cls_cfg, cls_path, role="classification")
            dual_summary["classification_model_file"] = str(cls_path)
            dual_summary["classification_choice"] = {
                "config": cls_result.config,
                "metrics": cls_result.test_metrics,
            }
            print(f"Exported classification model to: {cls_path}")

        dual_summary_path = root / "quick_recovery_dual_models.json"
        with dual_summary_path.open("w", encoding="utf-8") as f:
            json.dump(dual_summary, f, indent=2)
        print(f"Saved dual-model recipe to: {dual_summary_path}")


if __name__ == "__main__":
    main()
