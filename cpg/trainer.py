"""
Training pipeline for CPG-GNN with curriculum learning.
"""

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import tqdm

from .config import CPGConfig, TrainingConfig
from .model import CPGModel, create_cpg_model
from .graph_builder import CPGBuilder, CodePropertyGraph

logger = logging.getLogger(__name__)


@dataclass
class PackageSample:
    """Training sample representing a package."""
    package_name: str
    ecosystem: str
    label: int  # 0 = benign, 1 = malicious
    file_contents: Dict[str, str]
    sample_type: str = "unknown"  # benign, malicious_intent, compromised_lib
    metadata: Dict[str, Any] = field(default_factory=dict)


class CPGDataset(Dataset):
    """Dataset for CPG training."""
    
    def __init__(self, 
                 samples: List[PackageSample],
                 cpg_builder: CPGBuilder,
                 feature_extractor,
                 augment: bool = False,
                 augmentation_prob: float = 0.3):
        self.samples = samples
        self.cpg_builder = cpg_builder
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.augmentation_prob = augmentation_prob
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Apply augmentation if enabled
        file_contents = sample.file_contents
        if self.augment and random.random() < self.augmentation_prob:
            file_contents = self._apply_augmentation(file_contents)
        
        # Build CPG
        cpg = self.cpg_builder.build_package_graph(
            sample.package_name,
            sample.ecosystem,
            file_contents
        )
        
        # Extract features
        data = self.feature_extractor.extract_features(cpg)
        
        # Add label
        data.y = torch.tensor(sample.label, dtype=torch.long)
        
        # Add auxiliary labels
        data.api_labels = self._get_api_labels(cpg)
        data.entropy_label = self._get_entropy_label(file_contents)
        
        return data
    
    def _apply_augmentation(self, file_contents: Dict[str, str]) -> Dict[str, str]:
        """Apply data augmentation to file contents."""
        
        augmented = {}
        for file_path, content in file_contents.items():
            
            # Random augmentation choice
            aug_type = random.choice(['minification', 'variable_renaming', 'string_encoding'])
            
            if aug_type == 'minification' and file_path.endswith(('.js', '.py')):
                # Remove unnecessary whitespace
                lines = content.split('\n')
                minified_lines = [line.strip() for line in lines if line.strip()]
                augmented[file_path] = '\n'.join(minified_lines)
                
            elif aug_type == 'variable_renaming':
                # Simple variable name obfuscation
                content = self._obfuscate_variables(content)
                augmented[file_path] = content
                
            elif aug_type == 'string_encoding':
                # Encode some string literals
                content = self._encode_strings(content)
                augmented[file_path] = content
                
            else:
                augmented[file_path] = content
        
        return augmented
    
    def _obfuscate_variables(self, content: str) -> str:
        """Simple variable name obfuscation."""
        import re
        
        # Find variable patterns (simplified)
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        variables = set(re.findall(var_pattern, content))
        
        # Replace with obfuscated names
        for var in list(variables)[:10]:  # Limit to avoid breaking code
            if len(var) > 3 and var not in ['def', 'if', 'for', 'while', 'try', 'except']:
                obfuscated = f"_0x{hash(var) % 10000:04x}"
                content = content.replace(var, obfuscated)
        
        return content
    
    def _encode_strings(self, content: str) -> str:
        """Encode some string literals with base64."""
        import re
        import base64
        
        # Find string literals
        string_pattern = r'["\']([^"\']+)["\']'
        
        def encode_match(match):
            string_content = match.group(1)
            if len(string_content) > 5 and random.random() < 0.3:
                encoded = base64.b64encode(string_content.encode()).decode()
                return f'base64.b64decode("{encoded}").decode()'
            return match.group(0)
        
        return re.sub(string_pattern, encode_match, content)
    
    def _get_api_labels(self, cpg: CodePropertyGraph) -> torch.Tensor:
        """Get multi-label API presence vector."""
        labels = torch.zeros(len(self.cpg_builder.config.risky_apis))
        
        for i, api in enumerate(self.cpg_builder.config.risky_apis):
            if api in cpg.api_calls:
                labels[i] = 1.0
        
        return labels
    
    def _get_entropy_label(self, file_contents: Dict[str, str]) -> torch.Tensor:
        """Get entropy label for regression."""
        import math
        
        all_content = '\n'.join(file_contents.values())
        
        if not all_content:
            return torch.tensor(0.0)
        
        # Calculate Shannon entropy
        char_counts = {}
        for char in all_content:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        text_len = len(all_content)
        entropy = 0.0
        for count in char_counts.values():
            prob = count / text_len
            entropy -= prob * math.log2(prob)
        
        # Normalize
        max_entropy = math.log2(len(char_counts)) if char_counts else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return torch.tensor(normalized_entropy, dtype=torch.float32)


class CPGTrainer:
    """Trainer for CPG-GNN with curriculum learning."""
    
    def __init__(self, 
                 model: CPGModel,
                 config: TrainingConfig,
                 cpg_config: CPGConfig):
        self.model = model
        self.config = config
        self.cpg_config = cpg_config
        self.cpg_builder = CPGBuilder(cpg_config)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.api_loss = nn.BCEWithLogitsLoss()
        self.entropy_loss = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def train(self, train_samples: List[PackageSample], 
              val_samples: List[PackageSample]) -> Dict[str, Any]:
        """Train CPG-GNN with curriculum learning."""
        
        logger.info("Starting CPG-GNN training with curriculum learning")
        
        training_history = {
            "stages": [],
            "best_model_path": None,
            "final_metrics": {}
        }
        
        # Execute curriculum stages
        for stage_name, stage_config in self.config.curriculum_stages.items():
            logger.info(f"\n=== Training Stage: {stage_name} ===")
            
            # Prepare data for this stage
            stage_train_samples = self._prepare_stage_data(train_samples, stage_config)
            
            # Train this stage
            stage_history = self._train_stage(
                stage_name,
                stage_config,
                stage_train_samples,
                val_samples
            )
            
            training_history["stages"].append({
                "stage": stage_name,
                "history": stage_history
            })
            
            # Early stopping check
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at stage {stage_name}")
                break
        
        # Save final model
        final_model_path = self._save_checkpoint("final_model")
        training_history["best_model_path"] = final_model_path
        
        # Final evaluation
        final_metrics = self._evaluate_model(val_samples)
        training_history["final_metrics"] = final_metrics
        
        logger.info("Training completed!")
        return training_history
    
    def _train_stage(self, 
                    stage_name: str,
                    stage_config: Dict,
                    train_samples: List[PackageSample],
                    val_samples: List[PackageSample]) -> Dict[str, List]:
        """Train a single curriculum stage."""
        
        # Create datasets
        train_dataset = CPGDataset(
            train_samples,
            self.cpg_builder,
            self.model.feature_extractor,
            augment=stage_config.get("augmentation", False),
            augmentation_prob=self.config.augmentation_prob
        )
        
        val_dataset = CPGDataset(
            val_samples,
            self.cpg_builder,
            self.model.feature_extractor,
            augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_fn
        )
        
        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        
        max_epochs = stage_config.get("epochs", 20)
        
        for epoch in range(max_epochs):
            # Train
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            logger.info(
                f"Stage {stage_name}, Epoch {epoch+1}/{max_epochs}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )
            
            # Record history
            history["train_loss"].append(train_metrics['loss'])
            history["val_loss"].append(val_metrics['loss'])
            history["val_auc"].append(val_metrics['auc'])
            
            # Check for improvement
            current_score = val_metrics['auc']
            if current_score > self.best_score:
                self.best_score = current_score
                self.patience_counter = 0
                self._save_checkpoint(f"best_{stage_name}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            self.current_epoch += 1
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm.tqdm(dataloader, desc="Training"):
            batch = batch.to(self.model.device)
            
            # Forward pass
            output = self.model(batch)
            
            # Calculate losses
            classification_loss = self.classification_loss(output.logits, batch.y)
            
            # Auxiliary losses
            api_loss = 0.0
            entropy_loss = 0.0
            
            if hasattr(batch, 'api_labels') and output.api_predictions is not None:
                api_loss = self.api_loss(output.api_predictions.unsqueeze(0), batch.api_labels.unsqueeze(0))
            
            if hasattr(batch, 'entropy_label') and output.entropy_prediction is not None:
                entropy_loss = self.entropy_loss(output.entropy_prediction, batch.entropy_label.unsqueeze(0))
            
            # Combined loss
            total_loss_batch = (
                self.config.classification_loss_weight * classification_loss +
                self.config.api_prediction_loss_weight * api_loss +
                self.config.entropy_prediction_loss_weight * entropy_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_value
            )
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}
    
    def _validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.model.device)
                
                # Forward pass
                output = self.model(batch)
                
                # Calculate loss
                loss = self.classification_loss(output.logits, batch.y)
                total_loss += loss.item()
                
                # Collect predictions
                all_predictions.extend(output.probabilities[:, 1].cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                
                num_batches += 1
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if len(all_labels) > 0 and len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_predictions)
        else:
            auc = 0.5
        
        return {"loss": avg_loss, "auc": auc}
    
    def _prepare_stage_data(self, 
                           train_samples: List[PackageSample],
                           stage_config: Dict) -> List[PackageSample]:
        """Prepare training data for a curriculum stage."""
        
        malicious_ratio = stage_config.get("malicious_ratio", 0.5)
        include_trojans = stage_config.get("include_trojans", True)
        
        # Split samples by type
        benign_samples = [s for s in train_samples if s.label == 0]
        malicious_samples = [s for s in train_samples if s.label == 1]
        
        # Filter trojans if needed
        if not include_trojans:
            malicious_samples = [
                s for s in malicious_samples 
                if s.sample_type != "compromised_lib"
            ]
        
        # Calculate target counts
        total_malicious = len(malicious_samples)
        total_benign = int(total_malicious * (1 - malicious_ratio) / malicious_ratio)
        total_benign = min(total_benign, len(benign_samples))
        
        # Sample data
        selected_malicious = random.sample(malicious_samples, total_malicious)
        selected_benign = random.sample(benign_samples, total_benign)
        
        stage_samples = selected_malicious + selected_benign
        random.shuffle(stage_samples)
        
        logger.info(
            f"Stage data: {len(selected_malicious)} malicious, "
            f"{len(selected_benign)} benign "
            f"(ratio: {len(selected_malicious)/(len(selected_malicious)+len(selected_benign)):.2f})"
        )
        
        return stage_samples
    
    def _evaluate_model(self, test_samples: List[PackageSample]) -> Dict[str, float]:
        """Evaluate model on test set."""
        
        test_dataset = CPGDataset(
            test_samples,
            self.cpg_builder,
            self.model.feature_extractor,
            augment=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.model.device)
                output = self.model(batch)
                
                all_predictions.extend(output.prediction if isinstance(output.prediction, list) else [output.prediction])
                all_probabilities.extend(output.probabilities[:, 1].cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        if len(all_labels) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='binary'
            )
            
            if len(set(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_probabilities)
            else:
                auc = 0.5
            
            return {
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        return {"auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        if self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def _save_checkpoint(self, name: str) -> str:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{name}.pth")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_score': self.best_score,
            'cpg_config': self.cpg_config,
            'training_config': self.config
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def _collate_fn(self, batch):
        """Custom collate function for PyG data."""
        return Batch.from_data_list(batch)