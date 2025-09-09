"""
AMIL Training Pipeline with Curriculum Learning.
Implements 3-stage curriculum as specified in AMIL.md.
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import AMILConfig, TrainingConfig
from .model import AMILModel, create_amil_model
from .feature_extractor import AMILFeatureExtractor, UnitFeatures
from .losses import AMILLossFunction, create_loss_function, compute_loss_weights_from_data

logger = logging.getLogger(__name__)


@dataclass
class PackageSample:
    """Single package sample for training."""
    package_name: str
    label: int  # 0 = benign, 1 = malicious
    unit_features: List[UnitFeatures]
    ecosystem: str = "unknown"
    sample_type: str = "unknown"  # benign, compromised_lib, malicious_intent
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    """Training state tracking."""
    epoch: int = 0
    best_val_auc: float = 0.0
    best_model_path: Optional[str] = None
    patience_counter: int = 0
    total_training_time: float = 0.0
    stage_name: str = "stage_a"
    
    # Loss tracking
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_aucs: List[float] = field(default_factory=list)


class AMILDataset(Dataset):
    """Dataset for AMIL training with dynamic augmentation."""
    
    def __init__(self, samples: List[PackageSample], 
                 feature_extractor: AMILFeatureExtractor,
                 augmentation_config: Optional[Dict] = None,
                 max_units_per_package: int = 100):
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.augmentation_config = augmentation_config or {}
        self.max_units_per_package = max_units_per_package
        
        logger.info(f"Created AMIL dataset with {len(samples)} packages")
        self._log_dataset_stats()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, List[str]]:
        sample = self.samples[idx]
        
        # Apply augmentation if configured
        unit_features = self._maybe_augment_sample(sample.unit_features)
        
        # Limit number of units
        if len(unit_features) > self.max_units_per_package:
            # Sample top risky units (based on API features and entropy)
            unit_features = self._sample_top_units(unit_features)
        
        # Extract embeddings
        unit_embeddings = self.feature_extractor.forward(unit_features)
        
        # Get unit names
        unit_names = [f.unit_name for f in unit_features]
        
        return unit_embeddings, sample.label, unit_names
    
    def _log_dataset_stats(self):
        """Log dataset statistics."""
        labels = [s.label for s in self.samples]
        ecosystems = [s.ecosystem for s in self.samples]
        
        logger.info(f"  Benign: {labels.count(0)}, Malicious: {labels.count(1)}")
        logger.info(f"  Ecosystems: {dict(zip(*np.unique(ecosystems, return_counts=True)))}")
    
    def _maybe_augment_sample(self, unit_features: List[UnitFeatures]) -> List[UnitFeatures]:
        """Apply data augmentation if configured."""
        if not self.augmentation_config:
            return unit_features
        
        augmented_features = []
        for features in unit_features:
            # Apply different augmentation types based on probability
            augmented_content = features.raw_content
            
            # Minification
            if self._should_augment("minify"):
                augmented_content = self._apply_minification(augmented_content)
            
            # Base64 encoding of strings
            elif self._should_augment("base64"):
                augmented_content = self._apply_base64_augmentation(augmented_content)
            
            # String splitting
            elif self._should_augment("string_split"):
                augmented_content = self._apply_string_splitting(augmented_content)
            
            # Create new features with augmented content
            if augmented_content != features.raw_content:
                augmented_features.append(self._create_augmented_features(features, augmented_content))
            else:
                augmented_features.append(features)
        
        return augmented_features
    
    def _should_augment(self, aug_type: str) -> bool:
        """Check if should apply specific augmentation."""
        config = self.augmentation_config.get(aug_type, {})
        probability = config.get("probability", 0.0)
        return random.random() < probability
    
    def _apply_minification(self, content: str) -> str:
        """Apply minification augmentation."""
        config = self.augmentation_config.get("minify", {})
        
        if config.get("remove_comments", True):
            # Remove Python/JS comments
            import re
            content = re.sub(r'#.*', '', content)  # Python comments
            content = re.sub(r'//.*', '', content)  # JS comments
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # JS block comments
        
        if config.get("remove_whitespace", True):
            lines = content.split('\n')
            content = '\n'.join(line.strip() for line in lines if line.strip())
        
        return content
    
    def _apply_base64_augmentation(self, content: str) -> str:
        """Apply base64 encoding augmentation to string literals."""
        import re
        import base64
        
        def encode_string(match):
            string_content = match.group(1)
            if len(string_content) > 10:  # Only encode longer strings
                encoded = base64.b64encode(string_content.encode()).decode()
                return f'"__b64__{encoded}"'  # Marker for detection
            return match.group(0)
        
        # Encode string literals
        content = re.sub(r'["\']([^"\']{10,})["\']', encode_string, content)
        return content
    
    def _apply_string_splitting(self, content: str) -> str:
        """Apply string splitting augmentation."""
        import re
        
        config = self.augmentation_config.get("string_split", {})
        max_splits = config.get("max_splits", 3)
        min_length = config.get("min_length", 10)
        
        def split_string(match):
            string_content = match.group(1)
            if len(string_content) > min_length:
                # Split into random parts
                split_points = sorted(random.sample(range(1, len(string_content)), 
                                                  min(max_splits, len(string_content) - 1)))
                parts = []
                start = 0
                for point in split_points + [len(string_content)]:
                    parts.append(f'"{string_content[start:point]}"')
                    start = point
                return " + ".join(parts)
            return match.group(0)
        
        content = re.sub(r'["\']([^"\']{10,})["\']', split_string, content)
        return content
    
    def _create_augmented_features(self, original: UnitFeatures, augmented_content: str) -> UnitFeatures:
        """Create new UnitFeatures with augmented content."""
        return self.feature_extractor.extract_unit_features(
            raw_content=augmented_content,
            file_path=original.file_path,
            unit_name=f"{original.unit_name}_aug",
            unit_type=original.unit_type,
            ecosystem=original.ecosystem
        )
    
    def _sample_top_units(self, unit_features: List[UnitFeatures]) -> List[UnitFeatures]:
        """Sample top risky units when package is too large."""
        # Score units by risk indicators
        def risk_score(features):
            api_risk = sum(features.api_counts.get(api, 0) for api in 
                          ["subprocess.spawn", "eval.exec", "obfuscation.base64"])
            entropy_risk = features.shannon_entropy * features.obfuscation_score
            return api_risk + entropy_risk
        
        # Sort by risk score and take top units
        scored_features = [(risk_score(f), f) for f in unit_features]
        scored_features.sort(key=lambda x: x[0], reverse=True)
        
        return [f for _, f in scored_features[:self.max_units_per_package]]


class AMILTrainer:
    """
    AMIL training pipeline with 3-stage curriculum learning.
    
    Stage A: Balanced training (5:1 benign:malicious)
    Stage B: Add augmented variants 
    Stage C: Realistic ratio training (10:1)
    """
    
    def __init__(self, amil_config: AMILConfig, training_config: TrainingConfig,
                 model_name: str = "microsoft/graphcodebert-base"):
        self.amil_config = amil_config
        self.training_config = training_config
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create model and feature extractor
        self.model = create_amil_model(amil_config, self.device)
        self.feature_extractor = AMILFeatureExtractor(amil_config, model_name)
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # Training components (initialized in setup_training)
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        
        # Training state
        self.state = TrainingState()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, train_samples: List[PackageSample], 
              val_samples: Optional[List[PackageSample]] = None) -> Dict[str, Any]:
        """
        Main training loop with 3-stage curriculum.
        
        Args:
            train_samples: Training package samples
            val_samples: Optional validation samples (will split from train if not provided)
            
        Returns:
            Training history and final metrics
        """
        logger.info("🚀 Starting AMIL curriculum training")
        
        # Prepare data splits
        if val_samples is None:
            train_samples, val_samples = self._split_data(train_samples)
        
        # Setup training components
        self._setup_training(train_samples)
        
        training_history = {
            "stages": {},
            "total_time": 0.0,
            "final_metrics": {}
        }
        
        start_time = time.time()
        
        # Execute 3-stage curriculum
        for stage_name, stage_config in self.training_config.curriculum_stages.items():
            logger.info(f"📚 Starting {stage_name}: {stage_config['description']}")
            
            stage_history = self._train_stage(
                stage_name=stage_name,
                stage_config=stage_config,
                train_samples=train_samples,
                val_samples=val_samples
            )
            
            training_history["stages"][stage_name] = stage_history
        
        # Final evaluation
        final_metrics = self._final_evaluation(val_samples)
        training_history["final_metrics"] = final_metrics
        training_history["total_time"] = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_history['total_time']:.2f}s")
        logger.info(f"   Best validation AUC: {self.state.best_val_auc:.4f}")
        
        return training_history
    
    def _split_data(self, samples: List[PackageSample]) -> Tuple[List[PackageSample], List[PackageSample]]:
        """Split data into train/validation sets."""
        val_split = self.training_config.validation_split
        
        # Stratified split by label
        labels = [s.label for s in samples]
        indices = list(range(len(samples)))
        
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=val_split,
            stratify=labels,
            random_state=42
        )
        
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        
        logger.info(f"Data split: {len(train_samples)} train, {len(val_samples)} val")
        return train_samples, val_samples
    
    def _setup_training(self, train_samples: List[PackageSample]):
        """Setup optimizer, scheduler, and loss function."""
        
        # Compute class weights
        train_labels = torch.tensor([s.label for s in train_samples], dtype=torch.float32)
        loss_weights = compute_loss_weights_from_data(train_labels)
        
        # Update training config with computed weights
        self.training_config.sparsity_weight = loss_weights["sparsity_weight"]
        
        # Create optimizer
        if self.training_config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9
            )
        
        # Learning rate scheduler
        total_epochs = sum(stage["epochs"] for stage in self.training_config.curriculum_stages.values())
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs
        )
        
        # Loss function
        self.loss_function = create_loss_function(self.training_config, self.model)
        
        logger.info("Training setup complete")
        logger.info(f"  Optimizer: {self.training_config.optimizer}")
        logger.info(f"  Learning rate: {self.training_config.learning_rate}")
        logger.info(f"  Total epochs: {total_epochs}")
    
    def _train_stage(self, stage_name: str, stage_config: Dict, 
                    train_samples: List[PackageSample],
                    val_samples: List[PackageSample]) -> Dict[str, Any]:
        """Train single curriculum stage."""
        
        self.state.stage_name = stage_name
        
        # Prepare stage data with appropriate ratio
        stage_train_samples = self._prepare_stage_data(train_samples, stage_config)
        
        # Create data loaders
        train_dataset = AMILDataset(
            stage_train_samples,
            self.feature_extractor,
            stage_config.get("augmentation_config") if stage_config.get("augmentation") else None,
            self.amil_config.max_units_per_package
        )
        
        val_dataset = AMILDataset(
            val_samples,
            self.feature_extractor,
            max_units_per_package=self.amil_config.max_units_per_package
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # Training loop for this stage
        stage_history = {
            "train_losses": [],
            "val_losses": [],
            "val_aucs": [],
            "epochs": stage_config["epochs"]
        }
        
        for epoch in range(stage_config["epochs"]):
            # Training
            train_loss = self._train_epoch(train_loader, epoch)
            stage_history["train_losses"].append(train_loss)
            
            # Validation
            val_loss, val_auc = self._validate_epoch(val_loader)
            stage_history["val_losses"].append(val_loss)
            stage_history["val_aucs"].append(val_auc)
            
            # Learning rate update
            self.scheduler.step()
            
            # Early stopping and checkpointing
            self._handle_epoch_end(val_auc, epoch)
            
            # Logging
            if epoch % 5 == 0 or epoch == stage_config["epochs"] - 1:
                logger.info(f"  Epoch {epoch+1}/{stage_config['epochs']}: "
                           f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")
        
        logger.info(f"✅ Stage {stage_name} completed")
        return stage_history
    
    def _prepare_stage_data(self, train_samples: List[PackageSample], 
                           stage_config: Dict) -> List[PackageSample]:
        """Prepare training data with appropriate class ratio for stage."""
        
        # Separate by label
        benign_samples = [s for s in train_samples if s.label == 0]
        malicious_samples = [s for s in train_samples if s.label == 1]
        
        malicious_ratio = stage_config["malicious_ratio"]
        
        # Calculate sample counts
        num_malicious = len(malicious_samples)
        num_benign_needed = int(num_malicious * (1 - malicious_ratio) / malicious_ratio)
        
        # Sample benign data to achieve target ratio
        if len(benign_samples) > num_benign_needed:
            benign_samples = random.sample(benign_samples, num_benign_needed)
        
        stage_samples = benign_samples + malicious_samples
        random.shuffle(stage_samples)
        
        logger.info(f"  Stage data: {len(benign_samples)} benign, {len(malicious_samples)} malicious "
                   f"(ratio: {malicious_ratio:.2f})")
        
        return stage_samples
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train single epoch."""
        self.model.train()
        self.feature_extractor.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (unit_embeddings_batch, targets, unit_names_batch) in enumerate(train_loader):
            targets = targets.to(self.device).float()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss_dict = self.loss_function(unit_embeddings_batch, targets, unit_names_batch)
            loss = loss_dict["total_loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate single epoch."""
        self.model.eval()
        self.feature_extractor.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for unit_embeddings_batch, targets, unit_names_batch in val_loader:
                targets = targets.to(self.device).float()
                
                # Compute loss
                loss_dict = self.loss_function(unit_embeddings_batch, targets, unit_names_batch)
                total_loss += loss_dict["total_loss"].item()
                
                # Collect predictions
                for unit_embeddings in unit_embeddings_batch:
                    output = self.model(unit_embeddings, return_attention=False)
                    all_predictions.append(output.package_probability.cpu().numpy())
                
                all_targets.extend(targets.cpu().numpy())
        
        # Compute metrics
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets)
        
        val_auc = roc_auc_score(all_targets, all_predictions)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, val_auc
    
    def _handle_epoch_end(self, val_auc: float, epoch: int):
        """Handle end of epoch: early stopping, checkpointing."""
        
        # Update best model
        if val_auc > self.state.best_val_auc:
            self.state.best_val_auc = val_auc
            self.state.patience_counter = 0
            
            # Save best model
            checkpoint_path = self.checkpoint_dir / f"best_amil_model_{self.state.stage_name}.pt"
            self._save_checkpoint(checkpoint_path, is_best=True)
            self.state.best_model_path = str(checkpoint_path)
        else:
            self.state.patience_counter += 1
        
        # Regular checkpointing
        if epoch % self.training_config.save_every_n_epochs == 0:
            checkpoint_path = self.checkpoint_dir / f"amil_epoch_{epoch}_{self.state.stage_name}.pt"
            self._save_checkpoint(checkpoint_path, is_best=False)
        
        self.state.epoch += 1
    
    def _collate_fn(self, batch: List[Tuple]) -> Tuple[List[torch.Tensor], torch.Tensor, List[List[str]]]:
        """Custom collate function for variable-length packages."""
        unit_embeddings_batch = []
        targets = []
        unit_names_batch = []
        
        for unit_embeddings, target, unit_names in batch:
            unit_embeddings_batch.append(unit_embeddings.to(self.device))
            targets.append(target)
            unit_names_batch.append(unit_names)
        
        targets = torch.tensor(targets, dtype=torch.float32)
        
        return unit_embeddings_batch, targets, unit_names_batch
    
    def _save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "feature_extractor_state_dict": self.feature_extractor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state": self.state,
            "amil_config": self.amil_config,
            "training_config": self.training_config,
            "is_best": is_best
        }
        
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
            
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            self.state = checkpoint.get("training_state", TrainingState())
            
            logger.info(f"✅ Checkpoint loaded: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def _final_evaluation(self, val_samples: List[PackageSample]) -> Dict[str, float]:
        """Final comprehensive evaluation."""
        # Load best model
        if self.state.best_model_path:
            self.load_checkpoint(self.state.best_model_path)
        
        self.model.eval()
        self.feature_extractor.eval()
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for sample in val_samples:
                # Extract features and embeddings
                unit_embeddings = self.feature_extractor.forward(sample.unit_features)
                
                # Predict
                result = self.model.predict_package(unit_embeddings)
                
                all_predictions.append(1 if result["is_malicious"] else 0)
                all_targets.append(sample.label)
                all_confidences.append(result["confidence"])
        
        # Compute comprehensive metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        confidences = np.array(all_confidences)
        
        # Classification metrics
        roc_auc = roc_auc_score(targets, confidences)
        pr_auc = average_precision_score(targets, confidences)
        
        # Accuracy, precision, recall, F1
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)
        
        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_confidence": float(confidences.mean())
        }
        
        logger.info("📊 Final evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        return metrics


def create_trainer(amil_config: Optional[AMILConfig] = None,
                  training_config: Optional[TrainingConfig] = None) -> AMILTrainer:
    """Create AMIL trainer with default configs if not provided."""
    
    if amil_config is None:
        from .config import create_default_config
        amil_config, training_config, _ = create_default_config()
    
    if training_config is None:
        training_config = TrainingConfig()
    
    return AMILTrainer(amil_config, training_config)