import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import copy

# Dataset-specific channel statistics (from profile)
MEAN = np.array([0.740545, 0.53298219, 0.70582885], dtype=np.float32)
STD = np.array([0.12368222, 0.17676253, 0.12443067], dtype=np.float32)
N_CLASSES = 9


def _ensure_nchw(images):
    """Convert to NCHW format. Handles (N,H,W,C), (N,C,H,W), and edge cases."""
    if images.ndim != 4:
        raise ValueError(f"Expected 4D array, got {images.ndim}D with shape {images.shape}")
    s = images.shape
    # Already NCHW: (N, 3, 28, 28)
    if s[1] == 3 and s[2] == 28 and s[3] == 28:
        return images
    # NHWC: (N, 28, 28, 3)
    if s[1] == 28 and s[2] == 28 and s[3] == 3:
        return np.transpose(images, (0, 3, 1, 2))
    # Weird (N, H, C, W): (N, 28, 3, 28)
    if s[1] == 28 and s[2] == 3 and s[3] == 28:
        return np.transpose(images, (0, 2, 1, 3))
    # Fallback: last dim is channels
    if s[-1] == 3:
        return np.transpose(images, (0, 3, 1, 2))
    if s[1] == 3:
        return images
    raise ValueError(f"Cannot determine image format for shape {s}")


def _normalize(images):
    """Normalize NCHW images using dataset-specific stats. images: (N, 3, 28, 28)."""
    mean = MEAN.reshape(1, 3, 1, 1)
    std = STD.reshape(1, 3, 1, 1)
    return (images - mean) / std


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
        out += residual
        return F.relu(out, inplace=True)


class CompactResNet(nn.Module):
    """CIFAR-style ResNet for 28x28 RGB histopathology. ~3.2M params."""
    def __init__(self, n_classes=9, base_channels=64, num_blocks=2):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        # Stem: 3x3 conv stride 1, no maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        # Stage 1: c1 channels, 28x28
        self.stage1 = nn.Sequential(*[ResBlock(c1) for _ in range(num_blocks)])
        # Downsample to c2, 14x14
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.stage2 = nn.Sequential(*[ResBlock(c2) for _ in range(num_blocks)])
        # Downsample to c3, 7x7
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
        )
        self.stage3 = nn.Sequential(*[ResBlock(c3) for _ in range(num_blocks)])
        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(c3, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)       # (B, 64, 28, 28)
        x = self.stage1(x)    # (B, 64, 28, 28)
        x = self.down1(x)     # (B, 128, 28, 28)
        x = F.max_pool2d(x, 2)  # (B, 128, 14, 14)
        x = self.stage2(x)    # (B, 128, 14, 14)
        x = self.down2(x)     # (B, 256, 14, 14)
        x = F.max_pool2d(x, 2)  # (B, 256, 7, 7)
        x = self.stage3(x)    # (B, 256, 7, 7)
        x = self.pool(x).flatten(1)  # (B, 256)
        x = self.head(x)      # (B, 9)
        return x


def build_model(context):
    params = context.get('parameters', {})
    base_channels = params.get('base_channels', 64)
    num_blocks = params.get('num_blocks', 2)
    model = CompactResNet(n_classes=N_CLASSES, base_channels=base_channels, num_blocks=num_blocks)
    return model


def _get_continuation_identity(config):
    try:
        import autonomous
        return autonomous.continuation_identity(config)
    except Exception:
        return json.dumps({k: config.get(k) for k in ['seed', 'device', 'batch_size', 'lr']}, sort_keys=True)


def _get_data_identity(data):
    try:
        import adaptive_training
        return adaptive_training.data_identity(data)
    except Exception:
        n_train = data['train_images'].shape[0] if 'train_images' in data else 0
        return f"train_{n_train}"


def train(context):
    config = context['config']
    data = context['data']
    output_dir = Path(context['output'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Force CPU to avoid GPU OOM (another process uses ~19.7 GiB)
    device = torch.device('cpu')

    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = config.get('batch_size', 256)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    label_smoothing = config.get('label_smoothing', 0.1)
    grad_clip = config.get('gradient_clip_val', 1.0)
    es_patience = config.get('early_stopping_patience', 5)
    es_min_delta = config.get('early_stopping_min_delta', 0.001)
    epochs_target = context.get('epochs', 30)

    # Build model
    model = build_model(context)

    # Prepare data (CPU, float32)
    train_images = _ensure_nchw(data['train_images']).astype(np.float32)
    train_targets = data['train_targets'].astype(np.int64)
    val_images = _ensure_nchw(data['val_images']).astype(np.float32)
    val_targets = data['val_targets'].astype(np.int64)

    train_images = _normalize(train_images)
    val_images = _normalize(val_images)

    n_train = len(train_images)
    n_val = len(val_images)

    # Class weights (inverse frequency)
    class_counts = np.bincount(train_targets, minlength=N_CLASSES).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    max_count = class_counts.max()
    class_weights = (max_count / class_counts).astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights, device=device)

    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8
    )

    # Scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs_target, 1), eta_min=0.0)

    # Resume state
    resume = context.get('resume')
    start_epoch = 0
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    stale = 0
    stopped = False

    if resume is not None:
        start_epoch = resume.get('epochs_completed', 0)
        history = resume.get('history', [])
        best_val_acc = resume.get('best', 0.0)
        best_epoch = resume.get('best_epoch', 0)
        stale = resume.get('stale', 0)
        stopped = resume.get('stopped', False)
        # Load model
        model.load_state_dict(resume['model'])
        # Load optimizer
        optimizer.load_state_dict(resume['optimizer'])
        # Load scheduler
        if resume.get('scheduler') is not None:
            scheduler.load_state_dict(resume['scheduler'])
        # Restore RNG
        if resume.get('rng') is not None:
            torch.manual_seed(resume['rng'])
        if resume.get('loader_rng') is not None:
            np.random.seed(resume['loader_rng'])

    # Training loop
    for epoch in range(start_epoch + 1, epochs_target + 1):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.tensor(train_images[idx], device=device)
            yb = torch.tensor(train_targets[idx], device=device)

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
            val_logits = model(torch.tensor(val_images, device=device))
            val_preds = val_logits.argmax(dim=1).numpy()
            val_acc = float((val_preds == val_targets).mean())

        # Record history
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_accuracy': val_acc,
        })

        # Track best
        if val_acc > best_val_acc + es_min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            stale = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch}, output_dir / 'model.ckpt')
        else:
            stale += 1

        # Early stopping check
        if stale >= es_patience:
            stopped = True
            break

        # Step scheduler
        scheduler.step()

    epochs_completed = len(history)

    # Ensure model.ckpt exists
    if not (output_dir / 'model.ckpt').exists():
        torch.save({'model': model.state_dict(), 'epoch': epochs_completed}, output_dir / 'model.ckpt')

    # Save resume state
    np_state = np.random.get_state()
    resume_state = {
        'format': 'agent-resume-v1',
        'identity': _get_continuation_identity(config),
        'data_identity': _get_data_identity(data),
        'epochs_completed': epochs_completed,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng': torch.initial_seed(),
        'loader_rng': int(np_state[1][0]) if np_state[1].size > 0 else None,
        'history': history,
        'best': best_val_acc,
        'best_epoch': best_epoch,
        'stale': stale,
        'stopped': stopped,
    }
    torch.save(resume_state, output_dir / 'resume.pt')

    stop_reason = 'early_stopping' if stopped else 'segment_complete'

    return {
        'final_train_loss': history[-1]['train_loss'] if history else 0.0,
        'final_val_accuracy': history[-1]['val_accuracy'] if history else 0.0,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'epochs_completed': epochs_completed,
        'history': history,
        'stop_reason': stop_reason,
    }


def predict(context):
    checkpoint_path = context['checkpoint']
    device = torch.device('cpu')

    model = build_model(context)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    images = context['data']['images']
    images = _ensure_nchw(images).astype(np.float32)
    images = _normalize(images)

    batch_size = 512
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            xb = torch.tensor(images[i:i + batch_size], device=device)
            logits = model(xb)
            all_logits.append(logits.numpy())

    return np.concatenate(all_logits, axis=0)