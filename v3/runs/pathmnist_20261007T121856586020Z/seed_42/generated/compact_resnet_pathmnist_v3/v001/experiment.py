import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import json
import copy

# Dataset-specific normalization statistics
CHANNEL_MEAN = np.array([0.740545, 0.53298219, 0.70582885], dtype=np.float32)
CHANNEL_STD = np.array([0.12368222, 0.17676253, 0.12443067], dtype=np.float32)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CompactResNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # Stem: 3 -> 64, preserves 28x28
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Stage 1: 64 channels, 28x28
        self.stage1 = nn.Sequential(
            ResBlock(64, 64, stride=1),
            ResBlock(64, 64, stride=1)
        )
        # Stage 2: 64->128, 14x14
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1)
        )
        # Stage 3: 128->256, 7x7
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1)
        )
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def _normalize(images):
    """Normalize HWC uint8 or float images to NCHW float tensor with dataset stats."""
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    # HWC -> CHW -> NCHW
    images = images.transpose(0, 3, 1, 2).astype(np.float32)
    mean = CHANNEL_MEAN.reshape(1, 3, 1, 1)
    std = CHANNEL_STD.reshape(1, 3, 1, 1)
    images = (images - mean) / std
    return images


def _compute_class_weights(targets, num_classes=9):
    """Compute inverse-frequency class weights."""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.max()  # normalize so max weight = 1
    return weights


def build_model(context):
    """Build the model. Returns torch.nn.Module producing [batch, 9] logits."""
    config = context['config']
    params = context.get('parameters', {})
    num_classes = 9
    model = CompactResNet(num_classes=num_classes)
    return model


def train(context):
    """Train the model and return metrics."""
    config = context['config']
    data = context['data']
    output_dir = Path(context['output'])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    seed = config.get('seed', 42)
    batch_size = config.get('batch_size', 256)
    lr = config.get('lr', 2e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    label_smoothing = config.get('label_smoothing', 0.1)
    gradient_clip_val = config.get('gradient_clip_val', 1.0)
    early_stopping_patience = config.get('early_stopping_patience', 5)
    early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001)
    class_weighting = config.get('class_weighting', True)
    total_epochs = context.get('epochs', 30)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load data
    train_images = data['train_images']
    train_targets = data['train_targets']
    val_images = data['val_images']
    val_targets = data['val_targets']

    # Normalize
    train_x = _normalize(train_images)
    train_y = train_targets.astype(np.int64)
    val_x = _normalize(val_images)
    val_y = val_targets.astype(np.int64)

    # Create datasets and loaders (data stays on CPU, moved per-batch to GPU)
    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Build model
    model = CompactResNet(num_classes=9).to(device)

    # Class weights
    if class_weighting:
        cw = _compute_class_weights(train_y, 9)
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
    else:
        class_weights = None

    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.999), eps=1e-8)

    # Scheduler (cosine)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0.0)

    # Resume state
    start_epoch = 0
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    stale = 0
    stopped = False

    # Check for resume
    resume_path = output_dir / 'resume.pt'
    resume_state = None
    if resume_path.exists():
        try:
            resume_state = torch.load(resume_path, map_location='cpu', weights_only=False)
            if resume_state.get('format') == 'agent-resume-v1':
                model.load_state_dict(resume_state['model'])
                optimizer.load_state_dict(resume_state['optimizer'])
                if resume_state.get('scheduler') is not None:
                    scheduler.load_state_dict(resume_state['scheduler'])
                start_epoch = resume_state.get('epochs_completed', 0)
                history = resume_state.get('history', [])
                best_val_acc = resume_state.get('best', 0.0)
                best_epoch = resume_state.get('best_epoch', 0)
                stale = resume_state.get('stale', 0)
                stopped = resume_state.get('stopped', False)
                # Restore RNG
                if resume_state.get('rng') is not None:
                    torch.set_rng_state(resume_state['rng'])
                if resume_state.get('loader_rng') is not None:
                    np.random.set_state(resume_state['loader_rng'])
        except Exception:
            resume_state = None
            start_epoch = 0
            history = []
            best_val_acc = 0.0
            best_epoch = 0
            stale = 0
            stopped = False

    # Training loop
    final_train_loss = 0.0
    final_val_acc = 0.0

    for epoch in range(start_epoch + 1, total_epochs + 1):
        if stopped:
            break

        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        final_train_loss = avg_train_loss

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / max(total, 1)
        final_val_acc = val_acc

        # Step scheduler
        scheduler.step()

        # Track best
        if val_acc > best_val_acc + early_stopping_min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            stale = 0
            # Save best checkpoint
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_acc,
            }, output_dir / 'model.ckpt')
        else:
            stale += 1

        # Early stopping check
        if stale >= early_stopping_patience:
            stopped = True

        # Record history
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        # Save resume state
        resume_dict = {
            'format': 'agent-resume-v1',
            'identity': _get_identity(config),
            'data_identity': _get_data_identity(data),
            'epochs_completed': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'rng': torch.get_rng_state(),
            'loader_rng': np.random.get_state(),
            'history': history,
            'best': best_val_acc,
            'best_epoch': best_epoch,
            'stale': stale,
            'stopped': stopped,
        }
        torch.save(resume_dict, output_dir / 'resume.pt')

    # Ensure model.ckpt exists (save best if never updated)
    if not (output_dir / 'model.ckpt').exists():
        torch.save({
            'model': model.state_dict(),
            'epoch': total_epochs,
            'val_accuracy': final_val_acc,
        }, output_dir / 'model.ckpt')

    epochs_completed = len(history)
    stop_reason = 'early_stopping' if stopped else 'segment_complete'

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
    """Predict logits for inference images."""
    config = context['config']
    data = context['data']
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    images = data['images']
    x = _normalize(images)
    x_tensor = torch.from_numpy(x).to(device)

    # Load model from checkpoint
    ckpt_path = context.get('checkpoint')
    if ckpt_path is None:
        output_dir = Path(context['output'])
        ckpt_path = str(output_dir / 'model.ckpt')

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    model = CompactResNet(num_classes=9).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Batched inference
    batch_size = 512
    all_logits = []
    with torch.no_grad():
        for i in range(0, x_tensor.size(0), batch_size):
            xb = x_tensor[i:i+batch_size]
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def _get_identity(config):
    """Get continuation identity."""
    try:
        import autonomous
        return autonomous.continuation_identity(config)
    except (ImportError, Exception):
        return json.dumps({k: str(v) for k, v in config.items() if isinstance(v, (int, float, str, bool))}, sort_keys=True)


def _get_data_identity(data):
    """Get data identity."""
    try:
        import adaptive_training
        return adaptive_training.data_identity(data)
    except (ImportError, Exception):
        # Fallback: hash of data shapes
        parts = []
        for k in sorted(data.keys()):
            if isinstance(data[k], np.ndarray):
                parts.append(f"{k}:{data[k].shape}:{data[k].dtype}")
        return "|".join(parts)