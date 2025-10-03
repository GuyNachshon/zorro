"""
NeoBERT trainer implementation.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich import box

from .config import NeoBERTConfig, TrainingConfig
from .unit_processor import PackageUnit

logger = logging.getLogger(__name__)
console = Console()


class NeoBERTDataset(Dataset):
    """Dataset for NeoBERT training."""

    def __init__(self, samples: List[PackageUnit]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get label (default to 0.0 if not set)
        label = getattr(sample, 'malicious_label', 0.0)

        return {
            'unit_id': sample.unit_id,
            'token_ids': torch.tensor(sample.token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'risky_api_counts': sample.risky_api_counts,
            'entropy': sample.shannon_entropy,
            'phase': sample.phase_tag,
        }


def collate_fn(batch):
    """Collate function to handle variable-length sequences."""
    # Find max length in batch
    max_len = max(item['token_ids'].shape[0] for item in batch)

    # Pad sequences
    token_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        seq_len = item['token_ids'].shape[0]

        # Pad token_ids
        padded = F.pad(item['token_ids'], (0, max_len - seq_len), value=0)
        token_ids.append(padded)

        # Create attention mask
        mask = torch.zeros(max_len, dtype=torch.long)
        mask[:seq_len] = 1
        attention_masks.append(mask)

        # Label
        labels.append(item['label'])

    return {
        'token_ids': torch.stack(token_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels),
    }


class NeoBERTTrainer:
    """NeoBERT model trainer."""

    def __init__(self, model: nn.Module, training_config: TrainingConfig, neobert_config: NeoBERTConfig):
        self.model = model
        self.training_config = training_config
        self.neobert_config = neobert_config
        self.device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_samples: List[PackageUnit], val_samples: List[PackageUnit]) -> Dict[str, Any]:
        """Train the NeoBERT model."""
        logger.info("🎯 Starting NeoBERT training...")
        logger.info(f"Training samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(val_samples)}")
        logger.info(f"Device: {self.device}")

        # Create datasets
        train_dataset = NeoBERTDataset(train_samples)
        val_dataset = NeoBERTDataset(val_samples)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

        # Setup scheduler (simplified - just one stage for now)
        total_steps = len(train_loader) * 10  # 10 epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # Setup loss function with class weighting
        if self.training_config.use_class_weights:
            pos_weight = torch.tensor([self.training_config.pos_weight]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_auc = 0.0
        best_model_path = None
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': [],
        }

        num_epochs = 10  # Simplified - just one stage

        # Training header
        console.print(Panel.fit(
            "[bold cyan]Starting NeoBERT Training[/bold cyan]\n"
            f"Epochs: {num_epochs} | Batch Size: {self.training_config.batch_size} | "
            f"LR: {self.training_config.learning_rate:.0e}",
            border_style="cyan"
        ))

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            # Create progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Epoch {epoch+1}/{num_epochs} [Train]", total=len(train_loader))

                for batch_idx, batch in enumerate(train_loader):
                    labels = batch['labels'].to(self.device)

                    # Get PackageUnits from batch (stored in dataset)
                    # Since we can't pass units through DataLoader easily, we need to reconstruct
                    # For now, we'll extract the indices and get units from dataset
                    batch_start = batch_idx * self.training_config.batch_size
                    batch_end = min(batch_start + self.training_config.batch_size, len(train_samples))
                    batch_units = train_samples[batch_start:batch_end]

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(batch_units)

                    # Get logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits.squeeze(-1)
                    elif isinstance(outputs, dict):
                        logits = outputs['logits'].squeeze(-1)
                    else:
                        logits = outputs.squeeze(-1)

                    # Compute loss
                    loss = criterion(logits, labels)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clip_norm
                    )

                    optimizer.step()
                    scheduler.step()

                    # Track metrics
                    train_loss += loss.item()
                    train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())

                    progress.update(task, advance=1, description=f"[cyan]Epoch {epoch+1}/{num_epochs} [Train] Loss: {loss.item():.4f}")

            avg_train_loss = train_loss / len(train_loader)
            train_auc = roc_auc_score(train_labels, train_preds)

            # Validation phase
            val_loss, val_auc, val_f1, val_metrics = self.evaluate(val_loader, criterion, val_samples)

            # Save history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_f1'].append(val_f1)

            # Create metrics table
            metrics_table = Table(title=f"Epoch {epoch+1}/{num_epochs} Results", box=box.ROUNDED, show_header=True, header_style="bold magenta")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Train", justify="right")
            metrics_table.add_column("Val", justify="right")
            metrics_table.add_column("Best", justify="right", style="green")

            metrics_table.add_row("Loss", f"{avg_train_loss:.4f}", f"{val_loss:.4f}", f"{min(history['val_loss']):.4f}")
            metrics_table.add_row("AUC", f"{train_auc:.4f}", f"[bold]{val_auc:.4f}[/bold]", f"[bold green]{max(history['val_auc']):.4f}[/bold green]")
            metrics_table.add_row("F1", "-", f"{val_f1:.4f}", f"{max(history['val_f1']):.4f}")
            metrics_table.add_row("Precision", "-", f"{val_metrics['precision']:.4f}", "-")
            metrics_table.add_row("Recall", "-", f"{val_metrics['recall']:.4f}", "-")

            console.print(metrics_table)

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_path = "checkpoints/neobert/neobert_best.pth"

                # Create directory
                Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)

                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                }, best_model_path)

                console.print(f"[green]💾 Saved best model (AUC: {val_auc:.4f})[/green]\n")

        # Training complete
        results = {
            "training_completed": True,
            "best_model_path": best_model_path,
            "best_val_auc": best_val_auc,
            "final_train_loss": history['train_loss'][-1],
            "final_val_loss": history['val_loss'][-1],
            "final_val_auc": history['val_auc'][-1],
            "final_val_f1": history['val_f1'][-1],
            "epochs_completed": num_epochs,
            "history": history,
        }

        logger.info("✅ NeoBERT training completed!")
        logger.info(f"  Best Val AUC: {best_val_auc:.4f}")
        logger.info(f"  Final Val F1: {history['val_f1'][-1]:.4f}")

        return results

    def evaluate(self, dataloader, criterion, val_samples):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                labels = batch['labels'].to(self.device)

                # Get PackageUnits for this batch
                batch_start = batch_idx * self.training_config.batch_size
                batch_end = min(batch_start + self.training_config.batch_size, len(val_samples))
                batch_units = val_samples[batch_start:batch_end]

                # Forward pass
                outputs = self.model(batch_units)

                # Get logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits.squeeze(-1)
                elif isinstance(outputs, dict):
                    logits = outputs['logits'].squeeze(-1)
                else:
                    logits = outputs.squeeze(-1)

                # Compute loss
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # Store predictions
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)

        # Convert probabilities to binary predictions (threshold=0.5)
        binary_preds = (np.array(all_preds) > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, binary_preds, average='binary', zero_division=0
        )

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

        return avg_loss, auc, f1, metrics

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        logger.info(f"Loading NeoBERT checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"  Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"  Val AUC: {checkpoint.get('val_auc', 'unknown'):.4f}")