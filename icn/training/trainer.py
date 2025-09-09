"""
Base ICN Trainer with curriculum learning support.
Handles the complete training pipeline with 4-stage curriculum learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import time
import logging
from dataclasses import asdict
import os

# Import ICN components
from ..models.icn_model import ICNModel, ICNInput, ICNOutput
from ..training.losses import ICNLossComputer, BenignManifoldModel
from ..training.config import TrainingConfig, CurriculumConfig, CurriculumStageConfig
from ..training.wandb_config import ExperimentTracker, CurriculumStageTracker
from ..training.dataloader import create_icn_dataloader, CurriculumStage, ProcessedPackage
from ..evaluation.metrics import ICNMetrics
from ..data.data_preparation import ICNDataPreparator


class ICNTrainer:
    """
    Main trainer class for ICN with curriculum learning support.
    Handles the complete 4-stage training pipeline.
    """
    
    def __init__(
        self,
        model: ICNModel,
        config: TrainingConfig,
        train_packages: List[ProcessedPackage],
        eval_packages: Optional[List[ProcessedPackage]] = None,
        experiment_tracker: Optional[ExperimentTracker] = None
    ):
        self.model = model
        self.config = config
        self.train_packages = train_packages
        self.eval_packages = eval_packages or []
        
        # Initialize experiment tracking
        self.experiment_tracker = experiment_tracker
        self.curriculum_tracker = None
        if experiment_tracker:
            self.curriculum_tracker = CurriculumStageTracker(experiment_tracker)
        
        # Initialize curriculum
        self.curriculum_config = CurriculumConfig()
        self.current_stage = None
        self.current_stage_config = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.early_stopping_counter = 0
        
        # Components
        self.loss_computer = ICNLossComputer()
        self.benign_manifold = None
        self.metrics = ICNMetrics()
        
        # Device and distributed training
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Logging
        self.logger = self._setup_logging()
        
        # Checkpointing
        self.checkpoint_dir = Path(config.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ICN Trainer initialized")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Train packages: {len(self.train_packages)}")
        self.logger.info(f"  Eval packages: {len(self.eval_packages)}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        if device.type == "cuda":
            torch.cuda.set_device(device.index if device.index is not None else 0)
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("icn_trainer")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Console handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_optimizer(self, stage_config: CurriculumStageConfig) -> optim.Optimizer:
        """Create optimizer for the current stage."""
        
        # Group parameters with different learning rates
        param_groups = []
        
        # Encoder parameters (lower LR if pretrained)
        encoder_params = []
        if hasattr(self.model.local_estimator, 'encoder'):
            encoder_params = list(self.model.local_estimator.encoder.parameters())
        
        # Other model parameters
        other_params = [
            p for p in self.model.parameters() 
            if p not in encoder_params
        ]
        
        # Base learning rate from stage config
        base_lr = stage_config.learning_rate or self.config.optimizer.learning_rate
        
        if encoder_params:
            # Lower LR for encoder if using pretrained model
            encoder_lr = base_lr * 0.1 if self.config.use_pretrained else base_lr
            param_groups.append({
                'params': encoder_params,
                'lr': encoder_lr,
                'name': 'encoder'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other'
            })
        
        # Create optimizer
        if self.config.optimizer.type.value == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        elif self.config.optimizer.type.value == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=base_lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps
            )
        else:
            optimizer = optim.SGD(
                param_groups,
                lr=base_lr,
                weight_decay=self.config.optimizer.weight_decay,
                momentum=0.9
            )
        
        self.logger.info(f"Created optimizer: {type(optimizer).__name__}")
        self.logger.info(f"  Parameter groups: {len(param_groups)}")
        for i, group in enumerate(param_groups):
            self.logger.info(f"    Group {i} ({group['name']}): LR={group['lr']}, {len(group['params'])} params")
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int):
        """Create learning rate scheduler."""
        
        if self.config.scheduler.type.value == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.config.scheduler.warmup_steps
            )
        elif self.config.scheduler.type.value == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps
            )
        elif self.config.scheduler.type.value == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.config.scheduler.gamma
            )
        else:
            # No scheduler
            scheduler = None
        
        return scheduler
    
    def fit_benign_manifold(self):
        """Fit the benign manifold for plausibility detection."""
        self.logger.info("Fitting benign manifold...")
        
        benign_packages = [
            pkg for pkg in self.train_packages 
            if pkg.sample_type.value == "benign"
        ]
        
        if len(benign_packages) < 10:
            self.logger.warning("Not enough benign packages for manifold fitting")
            return
        
        # Extract embeddings from a subset of benign packages
        self.model.eval()
        benign_embeddings = []
        
        with torch.no_grad():
            for i, pkg in enumerate(benign_packages[:1000]):  # Limit for memory
                # Create a mini-batch
                input_ids_list = [pkg.input_ids]
                attention_masks_list = [pkg.attention_masks]
                phase_ids_list = [pkg.phase_ids]
                api_features_list = [pkg.api_features]
                ast_features_list = [pkg.ast_features]
                
                batch_input = ICNInput(
                    input_ids_list=input_ids_list,
                    attention_masks_list=attention_masks_list,
                    phase_ids_list=phase_ids_list,
                    api_features_list=api_features_list,
                    ast_features_list=ast_features_list,
                    manifest_embeddings=pkg.manifest_embedding.unsqueeze(0) if pkg.manifest_embedding is not None else None
                )
                
                # Forward pass to get global embedding
                output = self.model(batch_input)
                benign_embeddings.append(output.global_output.final_global_embedding[0])
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"  Processed {i + 1}/{len(benign_packages)} benign packages")
        
        if benign_embeddings:
            benign_embeddings = torch.stack(benign_embeddings)
            self.model.fit_benign_manifold(benign_embeddings)
            self.logger.info(f"Benign manifold fitted with {len(benign_embeddings)} samples")
        else:
            self.logger.error("No benign embeddings extracted")
    
    def train_stage(self, stage_name: str) -> Dict[str, float]:
        """Train a single curriculum stage."""
        
        stage_config = self.curriculum_config.get_stage_config(stage_name)
        self.current_stage = stage_name
        self.current_stage_config = stage_config
        
        self.logger.info(f"Starting {stage_config.name}")
        
        # Start stage tracking
        if self.curriculum_tracker:
            self.curriculum_tracker.start_stage(stage_name, asdict(stage_config))
        
        # Create stage-specific dataloader
        curriculum_stage = CurriculumStage(stage_name)
        train_dataloader = create_icn_dataloader(
            processed_packages=self.train_packages,
            batch_size=stage_config.batch_size or self.config.batch_size,
            curriculum_stage=curriculum_stage,
            balanced_sampling=stage_config.balanced_sampling,
            num_workers=self.config.dataloader_num_workers
        )
        
        eval_dataloader = None
        if self.eval_packages:
            eval_dataloader = create_icn_dataloader(
                processed_packages=self.eval_packages,
                batch_size=self.config.batch_size,
                curriculum_stage=curriculum_stage,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers
            )
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(stage_config)
        num_training_steps = len(train_dataloader) * stage_config.max_epochs
        scheduler = self.create_scheduler(optimizer, num_training_steps)
        
        # Update loss computer weights
        self.loss_computer.loss_weights.update(stage_config.loss_weights)
        
        # Training loop
        best_stage_metric = None
        stage_patience = 0
        
        for epoch in range(stage_config.max_epochs):
            
            # Training epoch
            train_metrics = self._train_epoch(
                train_dataloader, optimizer, scheduler, stage_config
            )
            
            # Evaluation epoch
            eval_metrics = {}
            if eval_dataloader:
                eval_metrics = self._eval_epoch(eval_dataloader, stage_config)
            
            # Combined metrics
            epoch_metrics = {**train_metrics, **eval_metrics}
            
            # Log metrics
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics(epoch_metrics, step=self.global_step)
            
            # Stage-specific metric tracking
            stage_metric_name = stage_config.eval_metric
            current_metric = epoch_metrics.get(f"eval_{stage_metric_name}", 
                                             epoch_metrics.get(stage_metric_name, 0.0))
            
            # Track best metric
            if best_stage_metric is None or current_metric > best_stage_metric:
                best_stage_metric = current_metric
                stage_patience = 0
                
                # Save best model for this stage
                self.save_checkpoint(f"best_{stage_name}")
            else:
                stage_patience += 1
            
            # Log stage metrics
            if self.curriculum_tracker:
                self.curriculum_tracker.log_stage_metric("epoch_metric", current_metric)
                self.curriculum_tracker.log_stage_metric("best_metric", best_stage_metric)
            
            self.logger.info(f"Epoch {epoch + 1}/{stage_config.max_epochs}")
            self.logger.info(f"  Train loss: {train_metrics.get('train_loss', 0.0):.4f}")
            if eval_metrics:
                self.logger.info(f"  Eval {stage_metric_name}: {current_metric:.4f}")
            self.logger.info(f"  Best {stage_metric_name}: {best_stage_metric:.4f}")
            
            # Early stopping for this stage
            if stage_patience >= stage_config.patience:
                self.logger.info(f"Early stopping for {stage_name} after {epoch + 1} epochs")
                break
            
            # Check stage completion criteria
            if self._check_stage_completion(stage_config, best_stage_metric):
                self.logger.info(f"Stage completion criteria met for {stage_name}")
                break
        
        # End stage tracking
        if self.curriculum_tracker:
            self.curriculum_tracker.end_stage()
        
        self.logger.info(f"Completed {stage_config.name}")
        self.logger.info(f"  Best {stage_config.eval_metric}: {best_stage_metric:.4f}")
        
        return {
            f"stage_{stage_name}_best_metric": best_stage_metric,
            f"stage_{stage_name}_final_metric": current_metric
        }
    
    def _train_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: optim.Optimizer, 
        scheduler, 
        stage_config: CurriculumStageConfig
    ) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast():
                    output = self.model(batch)
                    loss_dict = self._compute_losses(output, batch, stage_config)
                    loss = loss_dict["total"]
            else:
                output = self.model(batch)
                loss_dict = self._compute_losses(output, batch, stage_config)
                loss = loss_dict["total"]
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            
            # Update counters
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                
                step_metrics = {
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "global_step": self.global_step
                }
                
                # Add individual loss components
                for loss_name, loss_value in loss_dict.items():
                    if loss_value is not None and loss_name != "total":
                        step_metrics[f"train_{loss_name}"] = loss_value.item()
                
                if self.experiment_tracker:
                    self.experiment_tracker.log_metrics(step_metrics, step=self.global_step)
                
                self.logger.debug(f"Step {self.global_step}, Loss: {loss.item():.4f}")
        
        # Return epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss, "train_steps": num_batches}
    
    def _eval_epoch(self, dataloader: DataLoader, stage_config: CurriculumStageConfig) -> Dict[str, float]:
        """Evaluate for one epoch."""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                output = self.model(batch)
                loss_dict = self._compute_losses(output, batch, stage_config)
                
                total_loss += loss_dict["total"].item()
                
                # Collect predictions for metrics
                predictions = output.malicious_scores.cpu().numpy()
                labels = batch.malicious_labels.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # Compute metrics
        eval_metrics = self.metrics.compute_metrics(all_predictions, all_labels)
        eval_metrics["eval_loss"] = total_loss / len(dataloader)
        
        return eval_metrics
    
    def _compute_losses(
        self, 
        output: ICNOutput, 
        batch: ICNInput, 
        stage_config: CurriculumStageConfig
    ) -> Dict[str, torch.Tensor]:
        """Compute losses based on active losses for the stage."""
        
        # Prepare inputs for loss computation
        sample_types = [getattr(SampleType, st.upper()) for st in batch.sample_types]
        
        # Compute all losses
        losses = self.loss_computer.compute_losses(
            local_outputs=[output.local_outputs[i] for i in range(len(batch.sample_types))],
            global_output=output.global_output.final_global_intent,
            global_embeddings=output.global_output.final_global_embedding,
            sample_types=sample_types,
            intent_labels=batch.intent_labels,
            malicious_labels=batch.malicious_labels,
            benign_manifold=self.model.detection_system.plausibility_detector.get_prototypes(self.device) if self.model.detection_system.plausibility_detector.fitted else None,
            convergence_history=[state.global_intent_dist for state in output.global_output.convergence_history]
        )
        
        # Filter losses based on stage configuration
        filtered_losses = {}
        for loss_name in stage_config.active_losses:
            if loss_name in losses and losses[loss_name] is not None:
                filtered_losses[loss_name] = losses[loss_name]
        
        # Compute total loss
        total_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss_value in filtered_losses.items():
            weight = stage_config.loss_weights.get(loss_name, 1.0)
            total_loss += weight * loss_value
        
        filtered_losses["total"] = total_loss
        return filtered_losses
    
    def _move_batch_to_device(self, batch: ICNInput) -> ICNInput:
        """Move batch tensors to the training device."""
        
        # Move list of tensors
        input_ids_list = []
        attention_masks_list = []
        phase_ids_list = []
        api_features_list = []
        ast_features_list = []
        
        for i in range(len(batch.input_ids_list)):
            input_ids_list.append(batch.input_ids_list[i].to(self.device))
            attention_masks_list.append(batch.attention_masks_list[i].to(self.device))
            phase_ids_list.append(batch.phase_ids_list[i].to(self.device))
            api_features_list.append(batch.api_features_list[i].to(self.device))
            ast_features_list.append(batch.ast_features_list[i].to(self.device))
        
        # Move single tensors
        manifest_embeddings = batch.manifest_embeddings.to(self.device) if batch.manifest_embeddings is not None else None
        intent_labels = batch.intent_labels.to(self.device) if batch.intent_labels is not None else None
        malicious_labels = batch.malicious_labels.to(self.device) if batch.malicious_labels is not None else None
        
        return ICNInput(
            input_ids_list=input_ids_list,
            attention_masks_list=attention_masks_list,
            phase_ids_list=phase_ids_list,
            api_features_list=api_features_list,
            ast_features_list=ast_features_list,
            manifest_embeddings=manifest_embeddings,
            sample_types=batch.sample_types,
            intent_labels=intent_labels,
            malicious_labels=malicious_labels
        )
    
    def _check_stage_completion(self, stage_config: CurriculumStageConfig, current_metric: float) -> bool:
        """Check if stage completion criteria are met."""
        
        if current_metric >= stage_config.success_threshold:
            return True
        
        # Additional criteria could be checked here
        return False
    
    def save_checkpoint(self, checkpoint_name: str = "latest"):
        """Save model checkpoint."""
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'current_stage': self.current_stage,
            'best_metric': self.best_metric
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save to W&B if available
        if self.experiment_tracker:
            self.experiment_tracker.save_model_artifact(
                checkpoint_path, 
                f"icn_{checkpoint_name}",
                metadata={
                    'global_step': self.global_step,
                    'stage': self.current_stage,
                    'metric': self.best_metric
                }
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.current_stage = checkpoint.get('current_stage')
        self.best_metric = checkpoint.get('best_metric')
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"  Global step: {self.global_step}")
        self.logger.info(f"  Current stage: {self.current_stage}")
    
    def train_curriculum(self) -> Dict[str, float]:
        """Train the complete curriculum learning pipeline."""
        
        self.logger.info("Starting ICN Curriculum Training")
        self.logger.info(f"Total training packages: {len(self.train_packages)}")
        self.logger.info(f"Total evaluation packages: {len(self.eval_packages)}")
        
        # Fit benign manifold before training
        self.fit_benign_manifold()
        
        curriculum_results = {}
        
        # Train each stage in sequence
        stage_names = self.curriculum_config.get_stage_names()
        
        for stage_idx, stage_name in enumerate(stage_names):
            self.logger.info(f"Stage {stage_idx + 1}/{len(stage_names)}: {stage_name}")
            
            try:
                stage_results = self.train_stage(stage_name)
                curriculum_results.update(stage_results)
                
                # Save stage checkpoint
                self.save_checkpoint(f"stage_{stage_name}_complete")
                
            except Exception as e:
                self.logger.error(f"Error in stage {stage_name}: {e}")
                raise
        
        # Final evaluation
        self.logger.info("Curriculum training complete!")
        
        return curriculum_results
    
    @staticmethod
    def prepare_training_dataset(
        malicious_dataset_path: str = "malicious-software-packages-dataset",
        max_malicious_samples: Optional[int] = None,
        target_benign_count: int = 5000,
        force_recompute: bool = False,
        train_test_split: float = 0.8
    ) -> Tuple[List[ProcessedPackage], List[ProcessedPackage]]:
        """
        Prepare complete ICN training dataset with train/test split.
        
        Args:
            malicious_dataset_path: Path to malicious dataset
            max_malicious_samples: Max malicious samples (for testing)
            target_benign_count: Target number of benign samples
            force_recompute: Force recomputation of cached data
            train_test_split: Fraction for training set
            
        Returns:
            Tuple of (train_packages, eval_packages)
        """
        logger = logging.getLogger(__name__)
        logger.info("🔧 Preparing ICN training dataset...")
        
        # Initialize data preparator
        preparator = ICNDataPreparator(malicious_dataset_path=malicious_dataset_path)
        
        # Prepare complete dataset
        all_packages = preparator.prepare_complete_dataset(
            max_malicious_samples=max_malicious_samples,
            target_benign_count=target_benign_count,
            force_recompute=force_recompute
        )
        
        if not all_packages:
            raise ValueError("No packages were successfully processed!")
        
        # Save dataset statistics
        preparator.save_dataset_statistics(all_packages)
        
        # Split into train/eval sets
        import random
        random.shuffle(all_packages)  # Randomize order
        
        split_idx = int(len(all_packages) * train_test_split)
        train_packages = all_packages[:split_idx]
        eval_packages = all_packages[split_idx:]
        
        logger.info(f"📊 Dataset split complete:")
        logger.info(f"   Training packages: {len(train_packages)}")
        logger.info(f"   Evaluation packages: {len(eval_packages)}")
        
        # Print category distribution for both sets
        for name, packages in [("Training", train_packages), ("Evaluation", eval_packages)]:
            categories = {}
            for pkg in packages:
                category = pkg.sample_type.value
                categories[category] = categories.get(category, 0) + 1
            
            logger.info(f"   {name} distribution:")
            total = len(packages)
            for category, count in categories.items():
                logger.info(f"     {category}: {count} ({count/total*100:.1f}%)")
        
        return train_packages, eval_packages


if __name__ == "__main__":
    # This would normally be in a separate training script
    print("🚀 ICN Trainer implementation complete!")
    print("Ready for full curriculum training pipeline.")
    
    # The trainer would be used like:
    # 1. Load processed packages
    # 2. Create ICN model
    # 3. Create trainer with config
    # 4. Run trainer.train_curriculum()