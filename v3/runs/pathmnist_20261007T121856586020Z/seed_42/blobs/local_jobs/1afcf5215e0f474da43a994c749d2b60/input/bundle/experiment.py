import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import copy

# Dataset-specific normalization (from profile)
CHANNEL_MEAN = np.array([0.740545, 0.53298219, 0.70582885], dtype=np.float32)
CHANNEL_STD = np.array([0.12368222, 0.17676253, 0.12443067], dtype=np.float32)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out, inplace=True)


class CompactResNet(nn.Module):
    def __init__(self, num_classes=9, base_channels=64, dropout=0.3):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock(c1), ResBlock(c1))
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(ResBlock(c2), ResBlock(c2))
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(ResBlock(c3), ResBlock(c3))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def _to_nchw(images: np.ndarray) -> np.ndarray:
    """Convert NHWC (N,H,W,C) to NCHW (N,C,H,W)."""
    if images.ndim == 4 and images.shape[-1] in (1, 3):
        return np.transpose(images, (0, 3, 1, 2))
    return images


def _normalize(images: np.ndarray) -> np.ndarray:
    """Normalize NHWC images using dataset-specific stats."""
    mean = CHANNEL_MEAN[None, None, None, :]  # (1,1,1,3)
    std = CHANNEL_STD[None, None, None, :]
    return (images - mean) / std


def _prepare_batch(images: np.ndarray, targets: np.ndarray, batch_idx: int, batch_size: int, device: str):
    """Get a normalized NCHW batch on device."""
    start = batch_idx * batch_size
    end = min(start + batch_size, len(images))
    imgs = images[start:end]
    tgts = targets[start:end]
    # Normalize in NHWC, then transpose to NCHW
    imgs = _normalize(imgs)
    imgs = _to_nchw(imgs)
    x = torch.from_numpy(imgs).float().to(device)
    y = torch.from_numpy(tgts).long().to(device)
    return x, y


def _compute_class_weights(targets: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights."""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize so mean weight = 1
    return torch.tensor(weights, dtype=torch.float32)


def build_model(context):
    config = context['config']
    params = context.get('parameters', {})
    base_channels = params.get('base_channels', 64)
    dropout = params.get('dropout', 0.3)
    num_classes = 9
    model = CompactResNet(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    return model


def train(context):
    config = context['config']
    data = context['data']
    params = context.get('parameters', {})
    output_dir = Path(context['output'])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 256)
    lr = config.get('lr', 2e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    label_smoothing = config.get('label_smoothing', 0.1)
    class_weighting = config.get('class_weighting', True)
    grad_clip = config.get('gradient_clip_val', 1.0)
    patience = config.get('early_stopping_patience', 5)
    min_delta = config.get('early_stopping_min_delta', 0.001)
    monitor = config.get('early_stopping_monitor', 'val_accuracy')
    total_epochs = context.get('epochs', 30)
    seed = config.get('seed', 42)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Resume state
    resume = context.get('resume', None)
    start_epoch = 0
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    stale = 0

    if resume is not None:
        start_epoch = resume.get('epochs_completed', 0)
        history = resume.get('history', [])
        best_val_acc = resume.get('best', {}).get('val_accuracy', 0.0)
        best_epoch = resume.get('best_epoch', 0)
        stale = resume.get('stale', 0)
        # Load model state
        model_state = resume.get('model', None)
        if model_state is not None:
            model.load_state_dict(model_state)
        # Load optimizer state
        opt_state = resume.get('optimizer', None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        # Load scheduler state
        sched_state = resume.get('scheduler', None)
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
        # Load RNG state
        rng_state = resume.get('rng', None)
        if rng_state is not None:
            torch.manual_seed(rng_state)
        # Load best state
        best_state = resume.get('best_state', None)
    else:
        model = build_model(context)

    model = model.to(device)

    # Data preparation (keep on CPU, normalize per-batch)
    train_images = data['train_images']  # NHWC numpy
    train_targets = data['train_targets'].astype(np.int64)
    val_images = data['val_images']
    val_targets = data['val_targets'].astype(np.int64)

    n_train = len(train_images)
    n_val = len(val_images)
    num_classes = 9

    # Class weights
    if class_weighting:
        cw = _compute_class_weights(train_targets, num_classes).to(device)
    else:
        cw = None

    # Loss
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8
    )

    # Scheduler (cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1), eta_min=0.0)

    # If resuming, restore states
    if resume is not None:
        model_state = resume.get('model', None)
        if model_state is not None:
            model.load_state_dict(model_state)
        opt_state = resume.get('optimizer', None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        sched_state = resume.get('scheduler', None)
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
        rng_state = resume.get('rng', None)
        if rng_state is not None:
            torch.manual_seed(rng_state)
        best_state = resume.get('best_state', None)

    # Training loop
    stop_reason = 'segment_complete'
    epochs_completed = start_epoch

    for epoch in range(start_epoch + 1, total_epochs + 1):
        model.train()
        # Shuffle indices
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            imgs = train_images[idx]
            tgts = train_targets[idx]
            imgs = _normalize(imgs)
            imgs = _to_nchw(imgs)
            x = torch.from_numpy(imgs).float().to(device)
            y = torch.from_numpy(tgts).long().to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(idx)
            n_batches += 1

        train_loss = total_loss / n_train

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                imgs = val_images[i:i + batch_size]
                tgts = val_targets[i:i + batch_size]
                imgs = _normalize(imgs)
                imgs = _to_nchw(imgs)
                x = torch.from_numpy(imgs).float().to(device)
                y = torch.from_numpy(tgts).long().to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(tgts)

        val_acc = val_correct / max(val_total, 1)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Track best
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'learning_rate': current_lr,
        })

        epochs_completed = epoch

        # Early stopping check
        if stale >= patience:
            stop_reason = 'early_stopping'
            break

    # Save best model
    if best_state is not None:
        torch.save(best_state, output_dir / 'model.ckpt')
    else:
        torch.save(model.state_dict(), output_dir / 'model.ckpt')

    # Save resume state
    try:
        from autonomous import continuation_identity
        identity = continuation_identity(config)
    except Exception:
        identity = {'seed': seed, 'device': device}

    try:
        from adaptive_training import data_identity
        d_id = data_identity(data)
    except Exception:
        d_id = {'n_train': n_train, 'n_val': n_val}

    resume_dict = {
        'format': 'agent-resume-v1',
        'identity': identity,
        'data_identity': d_id,
        'epochs_completed': epochs_completed,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng': torch.initial_seed(),
        'loader_rng': int(np.random.get_state()[1][0]) if len(np.random.get_state()[1]) > 0 else 0,
        'history': history,
        'best': {'val_accuracy': best_val_acc},
        'best_epoch': best_epoch,
        'best_state': best_state,
        'stale': stale,
        'stopped': stop_reason == 'early_stopping',
    }
    torch.save(resume_dict, output_dir / 'resume.pt')

    final_train_loss = history[-1]['train_loss'] if history else 0.0
    final_val_acc = history[-1]['val_accuracy'] if history else 0.0

    return {
        'final_train_loss': final_train_loss,
        'final_val_accuracy': final_val_acc,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'epochs_completed': epochs_completed,
        'history': history,
        'stop_reason': stop_reason,
    }


def predict(context):
    config = context['config']
    data = context['data']
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 256)

    images = data['images']  # NHWC numpy
    n = len(images)

    # Load model
    ckpt_path = context.get('checkpoint', None)
    if ckpt_path is None:
        output_dir = Path(context['output'])
        ckpt_path = str(output_dir / 'model.ckpt')

    params = context.get('parameters', {})
    base_channels = params.get('base_channels', 64)
    dropout = params.get('dropout', 0.3)
    model = CompactResNet(num_classes=9, base_channels=base_channels, dropout=dropout)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Predict in batches
    all_logits = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            imgs = images[i:i + batch_size]
            imgs = _normalize(imgs)
            imgs = _to_nchw(imgs)
            x = torch.from_numpy(imgs).float().to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)