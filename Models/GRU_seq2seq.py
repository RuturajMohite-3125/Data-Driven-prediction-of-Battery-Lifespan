
import json
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from featureExtrcation.processDatasets import (
    HUSTDataProcessor, DQDV_V_WINDOWS, N_DQDV_FEATS,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_HERE, "processed_hust_MIT_cache.pkl")
RAW_PKL_PATH = "/Users/ruturaj/Master-Thesis/Dataset/MIT"
SPLIT_FILE = os.path.join(_HERE, "mix_cell_split.json")
SPLIT_KEY_MAP = {"Training": "train", "Validation": "val", "Testing": "test"}

_raw_cycles = os.environ.get("EOL_CYCLES_TO_USE", "")
CYCLES_TO_USE = (
    [int(c) for c in _raw_cycles.split(",") if c.strip().isdigit()]
    if _raw_cycles.strip()
    else [1,10,50,100,150,200,250]
)
CLASS_NAMES = ["Fast", "Normal", "Slow"]
N_CLASSES = 3
MIN_EOL_CYCLES = 200

SEED = int(os.environ.get("EOL_SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
SHOW_PLOTS = os.environ.get("EOL_SHOW_PLOTS", "1") == "1"
SAVE_ARTIFACTS = os.environ.get("EOL_SAVE_ARTIFACTS", "1") == "1"


MODEL_CFG = {
    "hidden_size":  int(os.environ.get("EOL_HIDDEN_SIZE",  "128")),
    "num_layers":   int(os.environ.get("EOL_NUM_LAYERS",   "1")),
    "head_dropout": float(os.environ.get("EOL_HEAD_DROPOUT", "0.5")),
}
TRAIN_CFG = {
    "epochs":       int(os.environ.get("EOL_EPOCHS",       "120")),
    "batch_size":   int(os.environ.get("EOL_BATCH_SIZE",   "8")),
    "lr":           float(os.environ.get("EOL_LR",           "3e-4")),
    "weight_decay": float(os.environ.get("EOL_WEIGHT_DECAY", "5e-2")),
    "lambda_eol":   float(os.environ.get("EOL_LAMBDA_EOL",   "1.0")),
}
LOSS_TYPE = os.environ.get("EOL_LOSS", "smooth_l1")
SCHEDULER = os.environ.get("EOL_SCHEDULER", "cosine")


USE_LOG_TARGET = os.environ.get("EOL_LOG_TARGET", "1") == "1"


def to_target(eol):
    """Map raw EOL (tensor) into the model's training target space."""
    return torch.log(eol.clamp(min=1e-6)) if USE_LOG_TARGET else eol


def from_target(t):
    """Invert to_target: model space -> raw cycles (numpy)."""
    t = np.asarray(t, dtype=float)
    return np.exp(t) if USE_LOG_TARGET else t


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class GRUEOLPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, head_dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(torch.mean(gru_out, dim=2), dim=1)
        context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)
        return self.head(context).squeeze(-1)


def load_cache(cache_path=CACHE_PATH):
    if not os.path.exists(cache_path):
        print(f"{cache_path} not found - generating from raw pkl files...")
        processor = HUSTDataProcessor(RAW_PKL_PATH)
        return processor.process_and_extract(cache_path=cache_path)

    with open(cache_path, "rb") as f:
        cells = pickle.load(f)
    print(f"Loaded {len(cells)} cells from {cache_path}")
    return cells


def load_fixed_split():
    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(f"{SPLIT_FILE} not found - this script expects a curated split.")

    with open(SPLIT_FILE) as f:
        raw = json.load(f)

    split = {}
    for src, tgt in SPLIT_KEY_MAP.items():
        if src not in raw:
            raise KeyError(f"{SPLIT_FILE} missing required key '{src}'.")
        split[tgt] = list(raw[src])

    sets = {k: set(v) for k, v in split.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[a] & sets[b]
        if overlap:
            print(f"WARNING: cells in both {a} and {b}: {sorted(overlap)} -> kept only in {a}")
            split[b] = [n for n in split[b] if n not in sets[a]]
            sets[b] = set(split[b])
    return split


def labels_from_train_tertiles(eol_arr, train_eol=None):
    """Bottom third -> Fast, middle -> Normal, top third -> Slow.
    Thresholds are fixed to match MIT_transformer_EOL.py."""
    if train_eol is None:
        raise ValueError("train_eol is required to derive train-only tertiles.")

    train_eol = np.asarray(train_eol, dtype=float)
    q33 = float(np.percentile(train_eol, 33.3))
    q66 = float(np.percentile(train_eol, 66.7))
    y = np.full(len(eol_arr), 1, dtype=int)
    y[np.asarray(eol_arr) <= q33] = 0
    y[np.asarray(eol_arr) > q66] = 2
    return y, (q33, q66)


def _aligned_proba(clf, X):
    probs = clf.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES), dtype=np.float32)
    for c_idx, c_lab in enumerate(clf.classes_):
        out[:, int(c_lab)] = probs[:, c_idx]
    return out


def train_xgb_aging_classifier(X_scalar, y, idx_tr, idx_va, idx_te, seed=42):
    Xt = X_scalar[idx_tr]
    yt = y[idx_tr]
    Xv = X_scalar[idx_va]
    Xte = X_scalar[idx_te]

    class_counts = np.bincount(yt, minlength=N_CLASSES).astype(float)
    class_weights = np.zeros(N_CLASSES, dtype=np.float32)
    nonzero = class_counts > 0
    if np.any(nonzero):
        class_weights[nonzero] = (len(yt) / (N_CLASSES * class_counts[nonzero])) ** 1.2

    base_kwargs = dict(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        tree_method="hist",
        n_jobs=-1,
    )

    n_splits = min(5, int(np.min(np.bincount(yt, minlength=N_CLASSES))) or 1)
    n_splits = max(2, n_splits)
    oof_pred = np.full(len(yt), -1, dtype=int)
    oof_proba = np.zeros((len(yt), N_CLASSES), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (a, b) in enumerate(skf.split(Xt, yt)):
        clf = XGBClassifier(random_state=seed + fold, **base_kwargs)
        clf.fit(Xt[a], yt[a], sample_weight=class_weights[yt[a]],
                eval_set=[(Xt[b], yt[b])], verbose=False)
        oof_pred[b] = clf.predict(Xt[b])
        oof_proba[b] = _aligned_proba(clf, Xt[b])

    final_clf = XGBClassifier(random_state=seed, **base_kwargs)
    rng = np.random.default_rng(seed)
    es_idx = rng.choice(len(Xt), size=max(1, len(Xt) // 5), replace=False)
    tr_idx = np.setdiff1d(np.arange(len(Xt)), es_idx)
    final_clf.fit(Xt[tr_idx], yt[tr_idx], sample_weight=class_weights[yt[tr_idx]],
                  eval_set=[(Xt[es_idx], yt[es_idx])], verbose=False)

    proba_va = _aligned_proba(final_clf, Xv)
    proba_te = _aligned_proba(final_clf, Xte)
    pred_va = final_clf.predict(Xv)
    pred_te = final_clf.predict(Xte)
    return final_clf, oof_pred, oof_proba, pred_va, proba_va, pred_te, proba_te


def append_class_features(X, proba):
    proba_t = torch.tensor(proba, dtype=torch.float32)
    tile = proba_t.unsqueeze(1).expand(-1, X.size(1), -1)
    return torch.cat([X, tile], dim=-1)


def build_cell_tensors(cells_cache, cycles=CYCLES_TO_USE):
    n_features = cells_cache[0]["features"].shape[1]
    X_list, eol_list, names = [], [], []
    for cell in cells_cache:
        eol = cell["eol"]
        if eol < MIN_EOL_CYCLES:
            continue
        feats = cell["features"]
        n_cyc = feats.shape[0]
        seq = np.zeros((len(cycles), n_features), dtype=np.float32)
        for t, c in enumerate(cycles):
            if c < n_cyc:
                seq[t] = feats[c]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(seq)
        eol_list.append(float(eol))
        names.append(cell["cell_name"])

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    eol = torch.tensor(eol_list, dtype=torch.float32).unsqueeze(-1)
    return X, eol, names


def eol_accuracy(preds, tgts, band=0.15):
    preds = np.asarray(preds, dtype=float)
    tgts = np.asarray(tgts, dtype=float)
    denom = np.clip(np.abs(tgts), 1e-8, None)
    return float(np.mean(np.abs(preds - tgts) / denom <= band))


def fit_class_blend(preds_va, tgts_va, proba_va, class_means):
    expected_from_class = np.asarray(proba_va, dtype=float) @ np.asarray(class_means, dtype=float)
    alphas = np.linspace(0.0, 1.0, 51)
    best_alpha, best_mae, best_preds = 0.0, float("inf"), np.asarray(preds_va, dtype=float)
    for a in alphas:
        cand = (1.0 - a) * preds_va + a * expected_from_class
        mae = float(np.mean(np.abs(cand - tgts_va)))
        if mae < best_mae:
            best_alpha, best_mae, best_preds = float(a), mae, cand
    return best_alpha, best_mae, best_preds


def apply_class_blend(preds, proba, class_means, alpha):
    expected_from_class = np.asarray(proba, dtype=float) @ np.asarray(class_means, dtype=float)
    return (1.0 - alpha) * np.asarray(preds, dtype=float) + alpha * expected_from_class


def plot_results(preds, tgts, title="Model Evaluation on Test Set"):
    errors = np.abs(preds - tgts)
    residuals = preds - tgts
    mae = float(np.mean(errors))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    lo = min(tgts.min(), preds.min())
    hi = max(tgts.max(), preds.max())
    ax.scatter(tgts, preds, alpha=0.6, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="Perfect prediction")
    ax.fill_between([lo, hi], [lo - 40, hi - 40], [lo + 40, hi + 40], color="green", alpha=0.12, label="±40 band")
    ax.set_xlabel("True EOL (cycles)")
    ax.set_ylabel("Predicted EOL (cycles)")
    ax.set_title("Predictions vs Ground Truth")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(errors, bins=20, edgecolor="white", alpha=0.8)
    ax.axvline(mae, color="red", ls="--", lw=1.5, label=f"MAE: {mae:.1f}")
    ax.axvline(40, color="green", ls="--", lw=1.5, label="±40 threshold")
    ax.set_xlabel("Absolute Error (cycles)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(tgts, residuals, alpha=0.6, edgecolors="none")
    ax.axhline(0, color="black", lw=1)
    ax.axhline(40, color="green", ls="--", lw=1, alpha=0.7)
    ax.axhline(-40, color="green", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("True EOL (cycles)")
    ax.set_ylabel("Residuals (cycles)")
    ax.set_title("Residuals vs Ground Truth")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    sorted_err = np.sort(errors)
    cumul = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax.plot(sorted_err, cumul, lw=2)
    ax.axvline(40, color="green", ls="--", lw=1.5, label="±40 threshold")
    within_40 = float(np.mean(errors <= 40) * 100)
    ax.axhline(within_40, color="green", ls=":", lw=1, alpha=0.5)
    ax.axhline(90, color="green", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Absolute Error (cycles)")
    ax.set_ylabel("Cumulative %")
    ax.set_title("Cumulative Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_eol_splits(preds_tr, tgts_tr, preds_va, tgts_va, preds_te, tgts_te,
                     title="GRU: Predicted vs True EOL - Train / Val / Test"):
    def _metrics(tgts, preds):
        tgts = np.asarray(tgts, dtype=float)
        preds = np.asarray(preds, dtype=float)
        mae = float(np.mean(np.abs(preds - tgts)))
        rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
        ss_res = np.sum((preds - tgts) ** 2)
        ss_tot = np.sum((tgts - np.mean(tgts)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return mae, rmse, r2

    splits = [
        ("Train", np.asarray(tgts_tr), np.asarray(preds_tr), "^", "#2a78d6"),
        ("Val",   np.asarray(tgts_va), np.asarray(preds_va), "s", "#eda100"),
        ("Test",  np.asarray(tgts_te), np.asarray(preds_te), "o", "#e34948"),
    ]

    all_tgts = np.concatenate([s[1] for s in splits])
    all_preds = np.concatenate([s[2] for s in splits])
    lo = min(all_tgts.min(), all_preds.min())
    hi = max(all_tgts.max(), all_preds.max())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.7, label="Perfect prediction")

    for label, tgts, preds, marker, color in splits:
        mae, rmse, r2 = _metrics(tgts, preds)
        ax.scatter(
            tgts, preds, marker=marker, s=70, color=color,
            edgecolors="white", linewidths=0.6, alpha=0.85,
            label=f"{label} (MAE={mae:.1f}, RMSE={rmse:.1f}, R²={r2:.2f})",
        )

    ax.set_xlabel("True EOL (cycles)")
    ax.set_ylabel("Predicted EOL (cycles)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def train_model(X_train, y_train, X_val, y_val,
                epochs=None, batch_size=None, lr=None, weight_decay=None, lambda_eol=None,
                model_cfg=None, loss_type=None, scheduler=None):
    _mcfg  = model_cfg  if model_cfg  is not None else MODEL_CFG
    _ltype = loss_type  if loss_type  is not None else LOSS_TYPE
    _sched = scheduler  if scheduler  is not None else SCHEDULER
    epochs       = epochs       if epochs       is not None else TRAIN_CFG["epochs"]
    batch_size   = batch_size   if batch_size   is not None else TRAIN_CFG["batch_size"]
    lr           = lr           if lr           is not None else TRAIN_CFG["lr"]
    weight_decay = weight_decay if weight_decay is not None else TRAIN_CFG["weight_decay"]
    lambda_eol   = lambda_eol   if lambda_eol   is not None else TRAIN_CFG["lambda_eol"]

    X_train = X_train.float()
    X_val   = X_val.float()
    y_train = y_train.float().squeeze(-1)
    y_val   = y_val.float().squeeze(-1)

    y_mean, y_std = y_train.mean().item(), y_train.std().item() + 1e-8
    y_train_n = (y_train - y_mean) / y_std
    y_val_n   = (y_val   - y_mean) / y_std

    train_loader = DataLoader(TensorDataset(X_train, y_train_n), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val_n),   batch_size=batch_size)

    model = GRUEOLPredictor(
        input_size=X_train.size(-1),
        hidden_size=_mcfg["hidden_size"],
        num_layers=_mcfg["num_layers"],
        head_dropout=_mcfg.get("head_dropout", 0.3),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if _sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
        sched_per_batch = True
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        sched_per_batch = False
    loss_fn = nn.L1Loss() if _ltype == "l1" else nn.SmoothL1Loss()

    best_val_loss, best_state = float("inf"), None
    patience, no_improve = 80, 0
    history = {"train_mae": [], "val_mae": []}

    print("Epoch |   TrLoss |   TrAcc |   VaLoss |   VaAcc |       LR")
    print("-" * 58)

    for ep in range(epochs):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        tr_preds, tr_tgts = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb) * lambda_eol
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched_per_batch:
                sched.step()
            batch_size_now = yb.size(0)
            tr_loss_sum += loss.item() * batch_size_now
            tr_n += batch_size_now
            tr_preds.append((pred.detach().cpu().numpy() * y_std) + y_mean)
            tr_tgts.append((yb.detach().cpu().numpy() * y_std) + y_mean)

        tr_preds = np.concatenate(tr_preds)
        tr_tgts = np.concatenate(tr_tgts)

        model.eval()
        va_loss_sum, va_n = 0.0, 0
        va_preds, va_tgts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                batch_size_now = yb.size(0)
                va_loss_sum += (loss_fn(pred, yb) * lambda_eol).item() * batch_size_now
                va_n += batch_size_now
                va_preds.append((pred.cpu().numpy() * y_std) + y_mean)
                va_tgts.append((yb.cpu().numpy() * y_std) + y_mean)

        tr_loss = tr_loss_sum / max(1, tr_n)
        va_loss = va_loss_sum / max(1, va_n)
        tr_acc = eol_accuracy(tr_preds, tr_tgts)
        va_acc = eol_accuracy(np.concatenate(va_preds), np.concatenate(va_tgts))
        tr_mae = float(np.mean(np.abs(tr_preds - tr_tgts)))
        va_mae = float(np.mean(np.abs(np.concatenate(va_preds) - np.concatenate(va_tgts))))
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(va_mae)

        if not sched_per_batch:
            sched.step()

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        current_lr = sched.get_last_lr()[0] if not sched_per_batch else lr
        if (ep + 1) % 20 == 0 or (ep + 1) == epochs:
            print(
                f"{ep + 1:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
                f"{va_loss:8.4f} | {va_acc:7.4f} | {current_lr:8.2e}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {ep+1} — val loss did not improve for {patience} epochs")
            break

    model.load_state_dict(best_state)
    return model, (y_mean, y_std), history, val_loader, best_val_loss


def evaluate(model, loader, y_mean, y_std, split_name="Test"):
    model.eval()
    preds, tgts = [], []
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            pred_n = model(xb)
            loss_sum += loss_fn(pred_n, yb.to(DEVICE)).item() * yb.size(0)
            n += yb.size(0)
            p = pred_n.cpu().numpy() * y_std + y_mean
            t = yb.numpy() * y_std + y_mean
            preds.append(p); tgts.append(t)
    preds = np.concatenate(preds); tgts = np.concatenate(tgts)
    preds = from_target(preds); tgts = from_target(tgts)
    abs_err = np.abs(preds - tgts)
    denom = np.clip(np.abs(tgts), 1e-8, None)
    rel_err = abs_err / denom
    sq_rel_err = ((preds - tgts) / denom) ** 2
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    mae_pct = float(np.mean(rel_err) * 100.0)
    rmse_pct = float(np.sqrt(np.mean(sq_rel_err)) * 100.0)

    eol_acc_10 = eol_accuracy(preds, tgts)
    loss = loss_sum / max(1, n)
    ss_res = np.sum((preds - tgts) ** 2)
    ss_tot = np.sum((tgts - np.mean(tgts)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f"[Stage 2] {split_name} accuracy: {eol_acc_10 * 100:.2f}%")
    print(
        f"[Stage 2] {split_name} metrics: MAE={mae:.2f}, MAE%={mae_pct:.2f}, "
        f"RMSE={rmse:.2f}, RMSE%={rmse_pct:.2f}, R2={r2:.4f}"
    )
    return preds, tgts, {
        "loss": loss,
        "mae": mae,
        "r2": r2,
        "rmse": rmse,
        "mae_pct": mae_pct,
        "rmse_pct": rmse_pct,
        "eol_accuracy_10": eol_acc_10,
    }


def make_eval_loader(X, y, y_mean, y_std, batch_size=8):
    y_norm = (y.squeeze(-1) - y_mean) / y_std
    return DataLoader(TensorDataset(X, y_norm), batch_size=batch_size)


def dqdv_window_cols(window, n_features):
    """Per-cycle feature-column indices for a dQ/dV voltage-window selection.

    window : 'all' / None -> every column;
             'none'       -> only the non-dQ/dV features (+ SOH);
             int w        -> window w's 3 dQ/dV stats + all non-dQ/dV features.
    """
    if window is None or window == "all":
        return None
    other = list(range(N_DQDV_FEATS, n_features))
    if window == "none":
        return other
    w = int(window)
    if not (0 <= w < len(DQDV_V_WINDOWS)):
        raise ValueError(f"dQ/dV window {w} out of range 0..{len(DQDV_V_WINDOWS) - 1}")
    return [3 * w, 3 * w + 1, 3 * w + 2] + other


if __name__ == "__main__":
    cells_cache = load_cache()
    X_all, eol_all, names = build_cell_tensors(cells_cache)

    _win = os.environ.get("EOL_DQDV_WINDOW", "all")
    _cols = dqdv_window_cols(_win, X_all.size(-1))
    if _cols is not None:
        X_all = X_all[:, :, _cols]
    print(f"[dQ/dV window: {_win}] After EOL filter: {len(names)} cells | "
          f"seq shape {tuple(X_all.shape)}")

    split = load_fixed_split()
    name_to_idx = {n: i for i, n in enumerate(names)}

    def split_idx(split_names, tag):
        kept, missing = [], []
        for n in split_names:
            (kept if n in name_to_idx else missing).append(n)
        if missing:
            print(f"  [{tag}] missing from data (filtered or absent): {missing}")
        return torch.tensor([name_to_idx[n] for n in kept], dtype=torch.long)

    idx_tr = split_idx(split["train"], "train")
    idx_va = split_idx(split["val"], "val")
    idx_te = split_idx(split["test"], "test")
    in_split = set(split["train"]) | set(split["val"]) | set(split["test"])
    extra = [n for n in names if n not in in_split]
    if extra:
        print(f"  cells extracted but not listed in any split - ignored: {extra}")
    print(f"Split sizes: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")

    X_tr = X_all[idx_tr]
    X_va = X_all[idx_va]
    X_te = X_all[idx_te]
    eol_tr = eol_all[idx_tr]
    eol_va = eol_all[idx_va]
    eol_te = eol_all[idx_te]

    X_np = X_all.numpy()
    feat_mean   = X_np.mean(axis=1)
    feat_std    = X_np.std(axis=1)
    feat_slope  = X_np[:, -1, :] - X_np[:, 0, :]
    feat_min    = X_np.min(axis=1)
    feat_max    = X_np.max(axis=1)
    feat_q25    = np.percentile(X_np, 25, axis=1)
    feat_q75    = np.percentile(X_np, 75, axis=1)
    q           = max(1, X_np.shape[1] // 4)
    feat_early  = X_np[:, :q, :].mean(axis=1)
    feat_late   = X_np[:, -q:, :].mean(axis=1)
    feat_deltas = np.diff(X_np, axis=1).reshape(X_np.shape[0], -1)
    scalar_X = np.concatenate([
        feat_mean, feat_std, feat_slope,
        feat_min, feat_max, feat_q25, feat_q75,
        feat_early, feat_late, feat_deltas,
    ], axis=1).astype(np.float32)
    scalar_X = np.nan_to_num(scalar_X, nan=0.0, posinf=0.0, neginf=0.0)

    eol_np = eol_all.squeeze(-1).numpy()
    y_all, (q33, q66) = labels_from_train_tertiles(eol_np, eol_np[idx_tr.numpy()])
    print(
        f"\n[Stage 1] Tertile thresholds: Fast EOL<={q33:.0f}, "
        f"Normal ({q33:.0f},{q66:.0f}], Slow >{q66:.0f}"
    )
    for c, name in enumerate(CLASS_NAMES):
        print(
            f"  train {name:<6}: {(y_all[idx_tr.numpy()] == c).sum():3d}   "
            f"val: {(y_all[idx_va.numpy()] == c).sum():3d}   "
            f"test: {(y_all[idx_te.numpy()] == c).sum():3d}"
        )

    (_clf, oof_pred, oof_proba, pred_va, proba_va, pred_te, proba_te) = train_xgb_aging_classifier(
        scalar_X,
        y_all,
        idx_tr.numpy(),
        idx_va.numpy(),
        idx_te.numpy(),
    )

    print("\n[Stage 1] OOF train classification report:")
    print(
        classification_report(
            y_all[idx_tr.numpy()],
            oof_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print(f"[Stage 1] Train accuracy: {accuracy_score(y_all[idx_tr.numpy()], oof_pred) * 100:.2f}%")
    print("[Stage 1] Val classification report:")
    print(
        classification_report(
            y_all[idx_va.numpy()],
            pred_va,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print(f"[Stage 1] Val accuracy: {accuracy_score(y_all[idx_va.numpy()], pred_va) * 100:.2f}%")
    print("[Stage 1] Test classification report:")
    print(
        classification_report(
            y_all[idx_te.numpy()],
            pred_te,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print(f"[Stage 1] Test accuracy: {accuracy_score(y_all[idx_te.numpy()], pred_te) * 100:.2f}%")
    print("[Stage 1] Test confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_all[idx_te.numpy()], pred_te, labels=[0, 1, 2]))

    class_means = []
    y_tr = y_all[idx_tr.numpy()]
    eol_tr_np = eol_tr.squeeze(-1).numpy()
    global_eol_mean = float(np.mean(eol_tr_np))
    for c in range(N_CLASSES):
        mask = y_tr == c
        class_means.append(float(np.mean(eol_tr_np[mask])) if np.any(mask) else global_eol_mean)
    print(f"[Stage 1] Train class-mean EOL priors (Fast, Normal, Slow): {class_means}")

    flat_tr = X_tr.reshape(-1, X_tr.size(-1))
    mu = flat_tr.mean(dim=0, keepdim=True)
    sd = flat_tr.std(dim=0, keepdim=True) + 1e-8
    X_tr = torch.nan_to_num((X_tr - mu) / sd)
    X_va = torch.nan_to_num((X_va - mu) / sd)
    X_te = torch.nan_to_num((X_te - mu) / sd)

    X_tr = append_class_features(X_tr, oof_proba)
    X_va = append_class_features(X_va, proba_va)
    X_te = append_class_features(X_te, proba_te)

    print(f"\nTrain tensor: {tuple(X_tr.shape)} (per-cycle features + 3 aging-class probs) | device: {DEVICE}")

    tgt_space = "log(EOL)" if USE_LOG_TARGET else "EOL"
    model, (y_mean, y_std), history, val_loader, best_mae = train_model(
        X_tr,
        to_target(eol_tr),
        X_va,
        to_target(eol_va),
        model_cfg=MODEL_CFG,
        loss_type=LOSS_TYPE,
        scheduler=SCHEDULER,
        **TRAIN_CFG,
    )

    
    print(f"\nBest val loss (normalized {tgt_space}): {best_mae:.4f}")

    print("\n[Train set]")
    train_loader = make_eval_loader(X_tr, to_target(eol_tr), y_mean, y_std, batch_size=1)
    preds_tr, tgts_tr, train_metrics = evaluate(model, train_loader, y_mean, y_std, split_name="Train")

    print("\n[Validation set]")
    preds_va, tgts_va, val_metrics = evaluate(model, val_loader, y_mean, y_std, split_name="Val")

    blend_alpha, blend_val_mae, preds_va_blend = fit_class_blend(preds_va, tgts_va, proba_va, class_means)
    raw_val_mae = float(np.mean(np.abs(preds_va - tgts_va)))
    print(
        f"[Calibration] Validation blend alpha={blend_alpha:.2f} | "
        f"raw MAE={raw_val_mae:.2f} -> blended MAE={blend_val_mae:.2f}"
    )

    print("\n[Test set]")
    test_loader = make_eval_loader(X_te, to_target(eol_te), y_mean, y_std, batch_size=1)
    preds_te, tgts_te, test_metrics = evaluate(model, test_loader, y_mean, y_std, split_name="Test")
    preds_te_blend = apply_class_blend(preds_te, proba_te, class_means, blend_alpha)
    raw_test_mae = float(np.mean(np.abs(preds_te - tgts_te)))
    test_mae = float(np.mean(np.abs(preds_te_blend - tgts_te)))
    print(
        f"[Calibration] Test MAE raw={raw_test_mae:.2f} -> blended={test_mae:.2f} "
        f"(alpha={blend_alpha:.2f})"
    )

    print(f"\n{'cell':<10} {'true':>6} {'pred':>7} {'|err|':>7} {'true_cls':>9} {'pred_cls':>9}  probs[F,N,S]")
    idx_te_np = idx_te.numpy()
    for i, name in enumerate([names[j] for j in idx_te.tolist()]):
        tc = CLASS_NAMES[y_all[idx_te_np[i]]]
        pc = CLASS_NAMES[pred_te[i]]
        flag = "" if tc == pc else "   <-- misclassified"
        p = proba_te[i]
        print(
            f"{name:<10} {tgts_te[i]:6.0f} {preds_te_blend[i]:7.1f} "
            f"{abs(preds_te_blend[i] - tgts_te[i]):7.1f}  "
            f"{tc:>9} {pc:>9}  [{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}]{flag}"
        )

    plot_results(preds_te_blend, tgts_te, title="Model Evaluation on Test Set")
    plot_eol_splits(preds_tr, tgts_tr, preds_va_blend, tgts_va, preds_te_blend, tgts_te)

    if SAVE_ARTIFACTS:
        ckpt_path = "best_eol_gru_50_cycle_window.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_size": X_tr.size(-1),
                    **MODEL_CFG,
                },
                "feature_mu": mu,
                "feature_sd": sd,
                "y_mean": y_mean,
                "y_std": y_std,
                "best_val_loss": best_mae,
                "cells_train": [names[j] for j in idx_tr.tolist()],
                "cells_val": [names[j] for j in idx_va.tolist()],
                "cells_test": [names[j] for j in idx_te.tolist()],
                "tertile_thresholds": (q33, q66),
                "class_eol_means": class_means,
                "blend_alpha": blend_alpha,
                "raw_test_mae": raw_test_mae,
                "test_mae": test_mae,
            },
            ckpt_path,
        )
        _clf.get_booster().save_model(os.path.join(_HERE, "aging_classifier_gru_50_xgb.json"))
        print(f"Saved best model -> {ckpt_path}")
        print("Saved aging classifier -> aging_classifier_gru_50_xgb.json")