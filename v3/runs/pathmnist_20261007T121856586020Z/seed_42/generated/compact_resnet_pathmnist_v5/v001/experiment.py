import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import random

# Dataset-specific normalization (from profile channel stats)
MEAN = np.array([0.740545, 0.53298219, 0.70582885], dtype=np.float32)
STD = np.array([0.12368222, 0.17676253, 0.12443067], dtype=np.float32)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CompactResNet(nn.Module):
    def __init__(self, num_classes=9, base_ch=64, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResBlock(base_ch, base_ch),
            ResBlock(base_ch, base_ch),
        )
        self.layer2 = nn.Sequential(
            ResBlock(base_ch, base_ch * 2, stride=2),
            ResBlock(base_ch * 2, base_ch * 2),
        )
        self.layer3 = nn.Sequential(
            ResBlock(base_ch * 2, base_ch * 4, stride=2),
            ResBlock(base_ch * 4, base_ch * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(base_ch * 4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head(x)
        return x


def build_model(context):
    params = context.get('parameters', {})
    base_ch = params.get('base_channels', 64)
    dropout = params.get('dropout', 0.3)
    model = CompactResNet(num_classes=9, base_ch=base_ch, dropout=dropout)
    return model


def _prepare(images):
    """HWC (N,28,28,3) float -> NCHW (N,3,28,28) normalized float32 numpy."""
    x = images.astype(np.float32, copy=False)
    x = (x - MEAN) / STD
    x = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
    return x


def _get_device(config):
    dev = config.get('device', 'cuda')
    if dev == 'cuda' and not torch.cuda.is_available():
        dev = 'cpu'
    return torch.device(dev)


def _save_resume(out_dir, config, data, model, optimizer, scheduler,
                 epochs_completed, history, best_val_acc, best_epoch, stale, stopped):
    try:
        from autonomous import continuation_identity
        from adaptive_training import data_identity
        identity = continuation_identity(config)
        d_identity = data_identity(data)
    except Exception:
        identity = None
        d_identity = None

    state = {
        'format': 'agent-resume-v1',
        'identity': identity,
        'data_identity': d_identity,
        'epochs_completed': epochs_completed,
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
    torch.save(state, out_dir / 'resume.pt')


def train(context):
    config = context['config']
    data = context['data']
    out_dir = Path(context['output'])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(config)
    seed = config.get('seed', 42)
    batch_size = config.get('batch_size', 128)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.01)
    label_smoothing = config.get('label_smoothing', 0.1)
    grad_clip = config.get('gradient_clip_val', 1.0)
    patience = config.get('early_stopping_patience', 5)
    min_delta = config.get('early_stopping_min_delta', 0.001)
    epochs_target = context.get('epochs', 30)

    # Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Model
    model = build_model(context).to(device)

    # Data (keep on CPU as numpy)
    train_x = _prepare(data['train_images'])
    train_y = data['train_targets'].astype(np.int64)
    val_x = _prepare(data['val_images'])
    val_y = data['val_targets'].astype(np.int64)

    # Class weights (inverse frequency, normalized so max = 1)
    counts = np.bincount(train_y, minlength=9).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.max()
    cw = torch.tensor(weights, dtype=torch.float32, device=device)

    # Loss
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs_target, 1), eta_min=0.0,
    )

    # State
    start_epoch = 0
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    stale = 0
    stopped = False

    # Resume
    resume_path = context.get('resume')
    if resume_path is not None:
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if state.get('scheduler') is not None:
            scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epochs_completed']
        history = state['history']
        best_val_acc = state['best']
        best_epoch = state['best_epoch']
        stale = state.get('stale', 0)
        if state.get('rng') is not None:
            torch.set_rng_state(state['rng'])
        if state.get('loader_rng') is not None:
            np.random.set_state(state['loader_rng'])

    n_train = len(train_y)
    n_val = len(val_y)

    for epoch in range(start_epoch + 1, epochs_target + 1):
        model.train()
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(train_x[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(train_y[idx]).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                xb = torch.from_numpy(val_x[i:i + batch_size]).to(device, non_blocking=True)
                yb = torch.from_numpy(val_y[i:i + batch_size]).to(device, non_blocking=True)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
        val_acc = correct / max(n_val, 1)

        scheduler.step()

        # Track best
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            stale = 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }, out_dir / 'model.ckpt')
        else:
            stale += 1

        history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        # Save resume state every epoch
        _save_resume(out_dir, config, data, model, optimizer, scheduler,
                     epoch, history, best_val_acc, best_epoch, stale, False)

        # Early stopping
        if stale >= patience:
            stopped = True
            break

    # Final resume save
    _save_resume(out_dir, config, data, model, optimizer, scheduler,
                 len(history), history, best_val_acc, best_epoch, stale, stopped)

    # Ensure model.ckpt exists
    if not (out_dir / 'model.ckpt').exists():
        torch.save({
            'model': model.state_dict(),
            'epoch': len(history),
            'val_acc': best_val_acc,
        }, out_dir / 'model.ckpt')

    return {
        'final_train_loss': history[-1]['train_loss'] if history else 0.0,
        'final_val_accuracy': history[-1]['val_accuracy'] if history else 0.0,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'epochs_completed': len(history),
        'history': history,
        'stop_reason': 'early_stopping' if stopped else 'segment_complete',
    }


def predict(context):
    config = context['config']
    data = context['data']
    device = _get_device(config)

    model = build_model(context)
    ckpt = torch.load(context['checkpoint'], map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    images = _prepare(data['images'])
    logits_list = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(images), bs):
            xb = torch.from_numpy(images[i:i + bs]).to(device, non_blocking=True)
            logits_list.append(model(xb).cpu().numpy())

    return np.concatenate(logits_list, axis=0)