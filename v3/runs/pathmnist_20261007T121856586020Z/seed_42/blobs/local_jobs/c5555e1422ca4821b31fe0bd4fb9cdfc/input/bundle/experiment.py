import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import math
import json
import copy
import os

try:
    import autonomous
except ImportError:
    autonomous = None

try:
    import adaptive_training
except ImportError:
    adaptive_training = None

# Dataset-specific normalization (from profile)
CHANNEL_MEAN = [0.740545, 0.53298219, 0.70582885]
CHANNEL_STD = [0.12368222, 0.17676253, 0.12443067]


class ConvTokenizer(nn.Module):
    def __init__(self, in_channels=3, hidden=128, kernel_size=4, stride=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, stride=stride)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * mlp_ratio, hidden),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.dropout(attn_out)
        residual = x
        x = residual + self.mlp(self.norm2(x))
        return x


class CompactTransformer(nn.Module):
    def __init__(self, num_classes=9, hidden=128, depth=4, num_heads=4,
                 mlp_ratio=4, dropout=0.1, patch_size=4):
        super().__init__()
        self.tokenizer = ConvTokenizer(3, hidden, kernel_size=patch_size, stride=patch_size)
        num_tokens = (28 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tokenizer(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def _get_params(context):
    p = context.get('parameters', {})
    return {
        'hidden': p.get('hidden', 128),
        'depth': p.get('depth', 4),
        'num_heads': p.get('num_heads', 4),
        'mlp_ratio': p.get('mlp_ratio', 4),
        'dropout': p.get('dropout', 0.1),
        'patch_size': p.get('patch_size', 4),
    }


def _to_tensor(images, device):
    """Convert numpy HWC or CHW images to NCHW float tensor on device."""
    if isinstance(images, np.ndarray):
        if images.ndim == 4 and images.shape[-1] == 3:
            # HWC -> CHW
            images = images.transpose(0, 3, 1, 2)
        elif images.ndim == 4 and images.shape[1] == 3:
            pass  # already NCHW
        t = torch.from_numpy(images.astype(np.float32))
    else:
        t = images
    return t.to(device)


def _normalize(x, device):
    mean = torch.tensor(CHANNEL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CHANNEL_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _compute_class_weights(targets, num_classes=9):
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = num_classes / (len(targets) * counts / counts.sum())
    # Normalize so max weight is ~1.63x (mild imbalance)
    weights = weights / weights.max()
    return torch.tensor(weights, dtype=torch.float32)


def _macro_f1(y_true, y_pred, num_classes=9):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def build_model(context):
    config = context.get('config', {})
    device = config.get('device', 'cpu')
    params = _get_params(context)
    model = CompactTransformer(
        num_classes=9,
        hidden=params['hidden'],
        depth=params['depth'],
        num_heads=params['num_heads'],
        mlp_ratio=params['mlp_ratio'],
        dropout=params['dropout'],
        patch_size=params['patch_size'],
    )
    return model.to(device)


def train(context):
    config = context.get('config', {})
    device = config.get('device', 'cpu')
    seed = config.get('seed', 42)
    batch_size = config.get('batch_size', 256)
    lr = config.get('lr', 2e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    label_smoothing = config.get('label_smoothing', 0.1)
    class_weighting = config.get('class_weighting', True)
    grad_clip = config.get('gradient_clip_val', 1.0)
    es_patience = config.get('early_stopping_patience', 5)
    es_monitor = config.get('early_stopping_monitor', 'val_accuracy')
    es_min_delta = config.get('early_stopping_min_delta', 0.001)
    scheduler_type = config.get('scheduler', 'cosine')
    cosine_eta_min = config.get('cosine_eta_min', 0.0)
    adam_beta1 = config.get('adam_beta1', 0.9)
    adam_beta2 = config.get('adam_beta2', 0.999)
    optimizer_eps = config.get('optimizer_eps', 1e-8)

    # Cumulative epochs target
    total_epochs = context.get('epochs', config.get('epochs', 30))

    # Data
    data = context['data']
    train_images = data['train_images']
    train_targets = data['train_targets']
    val_images = data['val_images']
    val_targets = data['val_targets']

    # Convert to tensors
    train_x = _to_tensor(train_images, device)
    train_y = torch.from_numpy(train_targets.astype(np.int64)).to(device)
    val_x = _to_tensor(val_images, device)
    val_y = torch.from_numpy(val_targets.astype(np.int64)).to(device)

    # Normalize
    train_x = _normalize(train_x, device)
    val_x = _normalize(val_x, device)

    # Class weights
    if class_weighting:
        cw = _compute_class_weights(train_targets, 9).to(device)
    else:
        cw = None

    # Loss
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    # Model
    params = _get_params(context)
    model = CompactTransformer(
        num_classes=9,
        hidden=params['hidden'],
        depth=params['depth'],
        num_heads=params['num_heads'],
        mlp_ratio=params['mlp_ratio'],
        dropout=params['dropout'],
        patch_size=params['patch_size'],
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2), eps=optimizer_eps
    )

    # Scheduler
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=cosine_eta_min * lr)
    elif scheduler_type == 'one_cycle':
        pct_start = config.get('one_cycle_pct_start', 0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_epochs * (len(train_x) // batch_size),
            pct_start=pct_start
        )
    elif scheduler_type == 'reduce_on_plateau':
        pf = config.get('plateau_factor', 0.5)
        pp = config.get('plateau_patience', 1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=pf, patience=pp, mode='max')

    # Output dir
    out_dir = Path(context['output'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume state
    start_epoch = 0
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    stale = 0
    stopped = False

    # Try to load resume state
    resume_path = out_dir / 'resume.pt'
    if resume_path.exists():
        try:
            resume_state = torch.load(resume_path, map_location=device, weights_only=False)
            if resume_state.get('format') == 'agent-resume-v1':
                model.load_state_dict(resume_state['model'])
                optimizer.load_state_dict(resume_state['optimizer'])
                if scheduler is not None and resume_state.get('scheduler') is not None:
                    scheduler.load_state_dict(resume_state['scheduler'])
                if resume_state.get('rng') is not None:
                    torch.set_rng_state(resume_state['rng'])
                if resume_state.get('loader_rng') is not None:
                    np.random.set_state(resume_state['loader_rng'])
                history = resume_state.get('history', [])
                best_val_acc = resume_state.get('best', 0.0)
                best_epoch = resume_state.get('best_epoch', 0)
                best_state = resume_state.get('best_state', None)
                start_epoch = resume_state.get('epochs_completed', 0)
                stale = resume_state.get('stale', 0)
                stopped = resume_state.get('stopped', False)
        except Exception:
            pass  # Start fresh if resume fails

    # Training loop
    n_train = len(train_x)
    n_val = len(val_x)
    rng_state = np.random.RandomState(seed + start_epoch)

    for epoch in range(start_epoch + 1, total_epochs + 1):
        model.train()
        # Shuffle
        perm = rng_state.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = train_x[idx]
            yb = train_y[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = F.cross_entropy(val_logits, val_y, reduction='mean').item()
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_true = val_y.cpu().numpy()
            val_acc = float(np.mean(val_pred == val_true))
            val_f1 = _macro_f1(val_true, val_pred, 9)

        # Scheduler step (epoch-based)
        if scheduler is not None:
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_acc if es_monitor == 'val_accuracy' else val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # History entry
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'val_macro_f1': val_f1,
            'learning_rate': current_lr,
        })

        # Track best
        monitor_val = val_acc if es_monitor == 'val_accuracy' else (val_f1 if es_monitor == 'val_macro_f1' else -val_loss)
        if monitor_val > best_val_acc + es_min_delta:
            best_val_acc = monitor_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        # Early stopping
        if stale >= es_patience:
            stopped = True
            break

    # Save best model as model.ckpt
    if best_state is not None:
        torch.save(best_state, out_dir / 'model.ckpt')
    else:
        torch.save(model.state_dict(), out_dir / 'model.ckpt')

    # Save resume state
    identity = None
    data_identity = None
    if autonomous is not None:
        try:
            identity = autonomous.continuation_identity(config)
        except Exception:
            pass
    if adaptive_training is not None:
        try:
            data_identity = adaptive_training.data_identity(data)
        except Exception:
            pass

    resume_dict = {
        'format': 'agent-resume-v1',
        'identity': identity,
        'data_identity': data_identity,
        'epochs_completed': min(total_epochs, start_epoch + (len(history) - (len(history) - len(history)))),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'rng': torch.get_rng_state(),
        'loader_rng': np.random.get_state(),
        'history': history,
        'best': best_val_acc,
        'best_epoch': best_epoch,
        'best_state': best_state,
        'stale': stale,
        'stopped': stopped,
    }
    # Fix epochs_completed
    epochs_completed = history[-1]['epoch'] if history else 0
    resume_dict['epochs_completed'] = epochs_completed
    torch.save(resume_dict, out_dir / 'resume.pt')

    # Final metrics
    final_train_loss = history[-1]['train_loss'] if history else 0.0
    final_val_acc = history[-1]['val_accuracy'] if history else 0.0
    best_val_acc_final = max(h['val_accuracy'] for h in history) if history else 0.0
    best_epoch_final = max(history, key=lambda h: h['val_accuracy'])['epoch'] if history else 0

    stop_reason = 'early_stopping' if stopped else 'segment_complete'

    return {
        'final_train_loss': final_train_loss,
        'final_val_accuracy': final_val_acc,
        'best_val_accuracy': best_val_acc_final,
        'best_epoch': best_epoch_final,
        'epochs_completed': epochs_completed,
        'history': history,
        'stop_reason': stop_reason,
    }


def predict(context):
    config = context.get('config', {})
    device = config.get('device', 'cpu')
    checkpoint_path = context.get('checkpoint', None)
    if checkpoint_path is None:
        out_dir = Path(context.get('output', '.'))
        checkpoint_path = str(out_dir / 'model.ckpt')

    params = _get_params(context)
    model = CompactTransformer(
        num_classes=9,
        hidden=params['hidden'],
        depth=params['depth'],
        num_heads=params['num_heads'],
        mlp_ratio=params['mlp_ratio'],
        dropout=params['dropout'],
        patch_size=params['patch_size'],
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    images = context['data']['images']
    x = _to_tensor(images, device)
    x = _normalize(x, device)

    # Batched inference
    batch_size = 512
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size]
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)