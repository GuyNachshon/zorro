"""
Weights & Biases configuration and experiment tracking for ICN.
"""

import wandb
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases experiments."""
    
    # Project settings
    project: str = "icn-malware-detection"
    entity: Optional[str] = None  # W&B username/team
    group: Optional[str] = None   # Experiment group
    job_type: str = "training"    # training, evaluation, inference
    
    # Experiment metadata
    experiment_name: Optional[str] = None
    tags: List[str] = None
    notes: str = ""
    
    # Logging settings
    log_frequency: int = 10  # Log every N steps
    save_code: bool = True
    log_gradients: bool = False  # Can be expensive
    watch_model: bool = True
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ExperimentTracker:
    """Handles experiment tracking with Weights & Biases integration."""
    
    def __init__(self, config: WandbConfig, training_config: Dict[str, Any]):
        self.config = config
        self.training_config = training_config
        self.run = None
        self.step_count = 0
        
    def init_experiment(self, model: torch.nn.Module = None) -> wandb.run:
        """Initialize W&B experiment."""
        
        # Prepare experiment config
        experiment_config = {
            **self.training_config,
            "framework": "pytorch",
            "icn_version": "phase2",
            "architecture": "dual_detection_channels"
        }
        
        # Initialize W&B run
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            job_type=self.config.job_type,
            name=self.config.experiment_name,
            tags=self.config.tags,
            notes=self.config.notes,
            config=experiment_config,
            save_code=self.config.save_code
        )
        
        # Watch model if provided
        if model is not None and self.config.watch_model:
            wandb.watch(
                model, 
                log="all" if self.config.log_gradients else "parameters",
                log_freq=self.config.log_frequency
            )
        
        print(f"🔍 Experiment initialized: {self.run.name}")
        print(f"   URL: {self.run.url}")
        
        return self.run
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.run is None:
            raise RuntimeError("Experiment not initialized. Call init_experiment() first.")
        
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        self.run.log(metrics, step=step)
    
    def log_curriculum_stage(self, stage: str, stage_config: Dict[str, Any]):
        """Log curriculum learning stage information."""
        self.log_metrics({
            f"curriculum/stage": wandb.config.get("stage_mapping", {}).get(stage, 0),
            f"curriculum/stage_name": stage
        })
        
        # Log stage-specific config
        stage_metrics = {f"stage_config/{k}": v for k, v in stage_config.items() 
                        if isinstance(v, (int, float, bool, str))}
        self.log_metrics(stage_metrics)
        
        print(f"📚 Curriculum stage: {stage}")
    
    def log_convergence_analysis(self, convergence_stats: Dict[str, Any], prefix: str = "convergence"):
        """Log convergence analysis metrics."""
        convergence_metrics = {}
        
        for key, value in convergence_stats.items():
            if isinstance(value, (int, float)):
                convergence_metrics[f"{prefix}/{key}"] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                convergence_metrics[f"{prefix}/{key}"] = value.item()
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                # Log statistical summary for lists
                convergence_metrics[f"{prefix}/{key}_mean"] = sum(value) / len(value)
                convergence_metrics[f"{prefix}/{key}_std"] = torch.tensor(value).std().item()
        
        self.log_metrics(convergence_metrics)
    
    def log_detection_analysis(self, detection_stats: Dict[str, Any]):
        """Log dual detection channel analysis."""
        detection_metrics = {}
        
        # Divergence channel metrics
        if "divergence" in detection_stats:
            div_stats = detection_stats["divergence"]
            for key, value in div_stats.items():
                if isinstance(value, (int, float)):
                    detection_metrics[f"divergence/{key}"] = value
                elif isinstance(value, torch.Tensor):
                    detection_metrics[f"divergence/{key}"] = value.mean().item()
        
        # Plausibility channel metrics
        if "plausibility" in detection_stats:
            plaus_stats = detection_stats["plausibility"]
            for key, value in plaus_stats.items():
                if isinstance(value, (int, float)):
                    detection_metrics[f"plausibility/{key}"] = value
                elif isinstance(value, torch.Tensor):
                    detection_metrics[f"plausibility/{key}"] = value.mean().item()
        
        # Combined metrics
        if "final_scores" in detection_stats:
            scores = detection_stats["final_scores"]
            if isinstance(scores, torch.Tensor):
                detection_metrics["detection/mean_score"] = scores.mean().item()
                detection_metrics["detection/score_std"] = scores.std().item()
        
        self.log_metrics(detection_metrics)
    
    def log_dataset_stats(self, dataset_stats: Dict[str, int], split: str = "train"):
        """Log dataset composition statistics."""
        dataset_metrics = {}
        
        for category, count in dataset_stats.items():
            dataset_metrics[f"dataset_{split}/{category}"] = count
        
        # Calculate ratios
        total = sum(dataset_stats.values())
        for category, count in dataset_stats.items():
            dataset_metrics[f"dataset_{split}/{category}_ratio"] = count / total
        
        self.log_metrics(dataset_metrics)
        
        print(f"📊 {split.title()} dataset stats:")
        for category, count in dataset_stats.items():
            print(f"   {category}: {count:,} ({count/total*100:.1f}%)")
    
    def save_model_artifact(self, model_path: Path, model_name: str, metadata: Dict[str, Any] = None):
        """Save model as W&B artifact."""
        if self.run is None:
            raise RuntimeError("Experiment not initialized.")
        
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=f"ICN model checkpoint from {self.run.name}",
            metadata=metadata or {}
        )
        
        artifact.add_file(str(model_path))
        self.run.log_artifact(artifact)
        
        print(f"💾 Model artifact saved: {model_name}")
    
    def log_sample_predictions(self, samples: List[Dict[str, Any]], max_samples: int = 10):
        """Log sample predictions with explanations."""
        if self.run is None:
            return
        
        # Create a table for sample analysis
        columns = ["package", "prediction", "confidence", "ground_truth", "primary_channel", "explanation"]
        data = []
        
        for i, sample in enumerate(samples[:max_samples]):
            data.append([
                sample.get("package_name", f"sample_{i}"),
                sample.get("prediction", "unknown"),
                sample.get("confidence", 0.0),
                sample.get("ground_truth", "unknown"),
                sample.get("primary_channel", "unknown"),
                str(sample.get("explanation", {}))[:200] + "..."  # Truncate long explanations
            ])
        
        table = wandb.Table(columns=columns, data=data)
        self.log_metrics({"predictions/sample_analysis": table})
    
    def log_confusion_matrix(self, y_true: List[int], y_pred: List[int], class_names: List[str]):
        """Log confusion matrix visualization."""
        if self.run is None:
            return
        
        try:
            import sklearn.metrics as metrics
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Compute confusion matrix
            cm = metrics.confusion_matrix(y_true, y_pred)
            
            # Create visualization
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('ICN Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Log to W&B
            self.log_metrics({"evaluation/confusion_matrix": wandb.Image(plt)})
            plt.close()
            
        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available for confusion matrix visualization")
    
    def finish_experiment(self):
        """Finish the W&B experiment."""
        if self.run is not None:
            self.run.finish()
            print(f"✅ Experiment completed: {self.run.name}")
            self.run = None


class CurriculumStageTracker:
    """Tracks curriculum learning stages and transitions."""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.current_stage = None
        self.stage_start_step = 0
        self.stage_metrics = {}
        
        # Define stage mappings for W&B
        self.stage_mapping = {
            "stage_a_pretraining": 0,
            "stage_b_convergence": 1, 
            "stage_c_malicious": 2,
            "stage_d_robustness": 3
        }
    
    def start_stage(self, stage: str, config: Dict[str, Any]):
        """Start a new curriculum stage."""
        if self.current_stage:
            self.end_stage()
        
        self.current_stage = stage
        self.stage_start_step = self.tracker.step_count
        self.stage_metrics = {}
        
        # Log stage transition
        stage_config = {
            **config,
            "stage_mapping": self.stage_mapping
        }
        self.tracker.log_curriculum_stage(stage, stage_config)
    
    def log_stage_metric(self, key: str, value: float):
        """Log a metric for the current stage."""
        if self.current_stage:
            full_key = f"stage_{self.current_stage}/{key}"
            self.tracker.log_metrics({full_key: value})
            self.stage_metrics[key] = value
    
    def end_stage(self):
        """End the current curriculum stage."""
        if self.current_stage:
            # Log stage summary
            stage_duration = self.tracker.step_count - self.stage_start_step
            summary_metrics = {
                f"stage_summary/{self.current_stage}/duration_steps": stage_duration,
                f"stage_summary/{self.current_stage}/completed": 1
            }
            
            # Add final stage metrics
            for key, value in self.stage_metrics.items():
                summary_metrics[f"stage_summary/{self.current_stage}/final_{key}"] = value
            
            self.tracker.log_metrics(summary_metrics)
            print(f"📝 Completed stage: {self.current_stage} ({stage_duration} steps)")
            
            self.current_stage = None
            self.stage_start_step = 0
            self.stage_metrics = {}


def create_experiment_config(
    experiment_name: str,
    stage: str = "training",
    tags: List[str] = None,
    notes: str = "",
    group: str = None
) -> WandbConfig:
    """Create a standard W&B config for ICN experiments."""
    
    default_tags = ["icn", "malware-detection", "intent-convergence"]
    if tags:
        default_tags.extend(tags)
    
    return WandbConfig(
        project="icn-malware-detection",
        experiment_name=experiment_name,
        job_type=stage,
        tags=default_tags,
        notes=notes,
        group=group,
        log_frequency=50,  # Log every 50 steps
        save_code=True,
        watch_model=True
    )


if __name__ == "__main__":
    # Test the experiment tracking setup
    print("🧪 Testing W&B Integration...")
    
    # Create test config
    wandb_config = create_experiment_config(
        experiment_name="icn_test_run",
        stage="testing",
        tags=["test", "phase2"],
        notes="Testing W&B integration for ICN"
    )
    
    training_config = {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "embedding_dim": 768,
        "max_iterations": 6,
        "convergence_threshold": 0.01
    }
    
    # Initialize tracker
    tracker = ExperimentTracker(wandb_config, training_config)
    
    # Test without actually initializing W&B (since we don't want to create real experiments)
    print("✅ W&B configuration created successfully")
    print(f"   Project: {wandb_config.project}")
    print(f"   Experiment: {wandb_config.experiment_name}")
    print(f"   Tags: {wandb_config.tags}")
    
    # Test curriculum stage tracker
    stage_tracker = CurriculumStageTracker(tracker)
    print("✅ Curriculum stage tracker ready")
    
    print("\n🚀 W&B Integration: Ready for training!")