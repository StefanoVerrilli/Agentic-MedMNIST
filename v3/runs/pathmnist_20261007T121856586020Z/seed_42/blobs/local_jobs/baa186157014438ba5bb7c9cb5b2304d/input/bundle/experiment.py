import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import copy

# Dataset-specific normalization statistics (PathMNIST 28x28 RGB)
CHANNEL_MEAN = np.array([0.740545, 0.53298219, 0.70582885], dtype=np.float32)
CHANNEL_STD = np.array([0.12368222, 0.17676253, 0.12443067], dtype=np.float32)


def _normalize(images_t):
    """Normalize NCHW float tensor using dataset-specific channel stats."""
    mean = torch.tensor(CHANNEL_MEAN, dtype=torch.float32, device=images_t.device).view(1, 3, 1, 1)
    std = torch.tensor(CHANNEL_STD, dtype=torch.float32, device=images_t.device).view(1, 3, 1, 1)
    return (images_t - mean) / std


def build_model(context):
    """Build the compact_transformer model via lightning_components."""
    from lightning_components import build_network
    params = context.get('parameters', {})
    hidden = params.get('hidden', 256)
    depth = params.get('depth', 4)
    num_heads = params.get('num_heads', 4)
    patch_size = params.get('patch_size', 4)
    dropout = params.get('dropout', 0.1)
    
    model = build_network(
        'compact_transformer',
        num_classes=9,
        in_channels=3,
        image_size=28,
        hidden=hidden,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        dropout=dropout,
    )
    return model


def _compute_class_weights(targets, num_classes=9):
    """Compute inverse-frequency class weights, normalized so max weight = 1."""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.max()
    return weights


def _validate(model, val_images_t, val_targets_t, device, batch_size=512):
    """Compute validation accuracy."""
    model.eval()
    n_val = len(val_targets_t)
    correct = 0
    with torch.no_grad():
        for i in range(0, n_val, batch_size):
            imgs = _normalize(val_images_t[i:i+batch_size])
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == val_targets_t[i:i+batch_size]).sum().item()
    return correct / n_val


def train(context):
    """Train the model with full resume support."""
    import autonomous
    import adaptive_training
    
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
    grad_clip = config.get('gradient_clip_val', 1.0)
    patience = config.get('early_stopping_patience', 5)
    min_delta = config.get('early_stopping_min_delta', 0.001)
    class_weighting = config.get('class_weighting', True)
    
    epochs_target = context.get('epochs', 30)
    
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
    stale = 0
    
    if resume is not None:
        start_epoch = resume.get('epochs_completed', 0)
        history = resume.get('history', [])
        best_val_acc = resume.get('best', {}).get('val_accuracy', 0.0)
        best_epoch = resume.get('best_epoch', 0)
        stale = resume.get('stale', 0)
    
    # Build model
    model = build_model(context)
    
    if resume is not None and 'model' in resume:
        model.load_state_dict(resume['model'])
    
    model = model.to(device)
    
    # Data preparation
    train_images = data['train_images']  # NCHW float numpy
    train_targets = data['train_targets']  # 1D int numpy
    val_images = data['val_images']
    val_targets = data['val_targets']
    
    train_images_t = torch.from_numpy(np.ascontiguousarray(train_images)).float().to(device)
    train_targets_t = torch.from_numpy(np.ascontiguousarray(train_targets)).long().to(device)
    val_images_t = torch.from_numpy(np.ascontiguousarray(val_images)).float().to(device)
    val_targets_t = torch.from_numpy(np.ascontiguousarray(val_targets)).long().to(device)
    
    n_train = len(train_targets_t)
    n_val = len(val_targets_t)
    
    # Class weights
    if class_weighting:
        cw = _compute_class_weights(train_targets, 9)
        class_weights_t = torch.from_numpy(cw).float().to(device)
    else:
        class_weights_t = None
    
    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    if resume is not None and 'optimizer' in resume:
        optimizer.load_state_dict(resume['optimizer'])
    
    # Scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs_target, 1), eta_min=0.0
    )
    
    if resume is not None and 'scheduler' in resume:
        scheduler.load_state_dict(resume['scheduler'])
    
    # Training loop
    stop_reason = 'segment_complete'
    last_epoch = start_epoch
    
    for epoch in range(start_epoch + 1, epochs_target + 1):
        last_epoch = epoch
        model.train()
        
        # Shuffle indices
        perm = torch.randperm(n_train, device=device)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            imgs = _normalize(train_images_t[idx])
            tgts = train_targets_t[idx]
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, tgts)
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        
        # Validation
        val_acc = _validate(model, val_images_t, val_targets_t, device, batch_size=512)
        
        # Track best
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            stale = 0
            # Save best checkpoint
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_acc,
                'channel_mean': CHANNEL_MEAN.tolist(),
                'channel_std': CHANNEL_STD.tolist(),
            }, output_dir / 'model.ckpt')
        else:
            stale += 1
        
        # Record history
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })
        
        # Scheduler step
        scheduler.step()
        
        # Early stopping
        if stale >= patience:
            stop_reason = 'early_stopping'
            break
    
    epochs_completed = last_epoch
    
    # Save resume state
    resume_state = {
        'format': 'agent-resume-v1',
        'identity': autonomous.continuation_identity(config),
        'data_identity': adaptive_training.data_identity(data),
        'epochs_completed': epochs_completed,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng': torch.get_rng_state(),
        'loader_rng': None,
        'history': history,
        'best': {'val_accuracy': best_val_acc},
        'best_epoch': best_epoch,
        'stale': stale,
        'stopped': stop_reason == 'early_stopping',
    }
    torch.save(resume_state, output_dir / 'resume.pt')
    
    # Ensure model.ckpt exists (edge case: no improvement in this segment)
    if not (output_dir / 'model.ckpt').exists():
        torch.save({
            'model': model.state_dict(),
            'epoch': epochs_completed,
            'val_accuracy': best_val_acc,
            'channel_mean': CHANNEL_MEAN.tolist(),
            'channel_std': CHANNEL_STD.tolist(),
        }, output_dir / 'model.ckpt')
    
    final_train_loss = history[-1]['train_loss'] if history else 0.0
    final_val_accuracy = history[-1]['val_accuracy'] if history else 0.0
    
    return {
        'final_train_loss': final_train_loss,
        'final_val_accuracy': final_val_accuracy,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'epochs_completed': epochs_completed,
        'history': history,
        'stop_reason': stop_reason,
    }


def predict(context):
    """Load checkpoint and produce logits for inference images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = context['checkpoint']
    
    model = build_model(context)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    
    images = context['data']['images']  # NCHW float numpy
    images_t = torch.from_numpy(np.ascontiguousarray(images)).float().to(device)
    
    n = len(images_t)
    batch_size = 512
    all_logits = []
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            imgs = _normalize(images_t[i:i+batch_size])
            logits = model(imgs)
            all_logits.append(logits.cpu().numpy())
    
    return np.concatenate(all_logits, axis=0)