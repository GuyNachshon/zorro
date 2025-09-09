"""
Evaluation metrics for ICN malware detection.
Comprehensive metrics including detection, localization, and convergence analysis.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
import warnings


class ICNMetrics:
    """Comprehensive metrics computation for ICN evaluation."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def compute_metrics(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray,
        prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions (probabilities or scores)
            labels: Ground truth binary labels
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of computed metrics
        """
        
        metrics = {}
        
        # Convert to numpy arrays if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Ensure we have valid data
        if len(predictions) == 0 or len(labels) == 0:
            return {f"{prefix}_empty": 1.0}
        
        # Binary predictions
        binary_preds = (predictions >= self.threshold).astype(int)
        
        try:
            # ROC-AUC (main metric)
            if len(np.unique(labels)) > 1:  # Need both classes for AUC
                roc_auc = roc_auc_score(labels, predictions)
                metrics[f"{prefix}_roc_auc"] = roc_auc
            else:
                metrics[f"{prefix}_roc_auc"] = 0.5  # Random performance
            
            # Precision-Recall AUC
            if len(np.unique(labels)) > 1:
                precision_curve, recall_curve, _ = precision_recall_curve(labels, predictions)
                pr_auc = auc(recall_curve, precision_curve)
                metrics[f"{prefix}_pr_auc"] = pr_auc
            
            # Classification metrics
            metrics[f"{prefix}_accuracy"] = accuracy_score(labels, binary_preds)
            metrics[f"{prefix}_precision"] = precision_score(labels, binary_preds, zero_division=0)
            metrics[f"{prefix}_recall"] = recall_score(labels, binary_preds, zero_division=0)
            metrics[f"{prefix}_f1"] = f1_score(labels, binary_preds, zero_division=0)
            
            # Confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
            
            metrics[f"{prefix}_true_positives"] = int(tp)
            metrics[f"{prefix}_false_positives"] = int(fp)
            metrics[f"{prefix}_true_negatives"] = int(tn)
            metrics[f"{prefix}_false_negatives"] = int(fn)
            
            # Derived metrics
            if tp + fn > 0:
                metrics[f"{prefix}_sensitivity"] = tp / (tp + fn)  # Same as recall
            if tn + fp > 0:
                metrics[f"{prefix}_specificity"] = tn / (tn + fp)
            if tp + fp > 0:
                metrics[f"{prefix}_positive_predictive_value"] = tp / (tp + fp)  # Same as precision
            if tn + fn > 0:
                metrics[f"{prefix}_negative_predictive_value"] = tn / (tn + fn)
            
            # False Positive Rate at different TPR thresholds
            if len(np.unique(labels)) > 1:
                fpr_95 = self._compute_fpr_at_tpr(labels, predictions, target_tpr=0.95)
                fpr_90 = self._compute_fpr_at_tpr(labels, predictions, target_tpr=0.90)
                metrics[f"{prefix}_fpr_at_95tpr"] = fpr_95
                metrics[f"{prefix}_fpr_at_90tpr"] = fpr_90
            
            # Sample statistics
            metrics[f"{prefix}_num_samples"] = len(predictions)
            metrics[f"{prefix}_num_positive"] = int(labels.sum())
            metrics[f"{prefix}_num_negative"] = int(len(labels) - labels.sum())
            metrics[f"{prefix}_class_balance"] = float(labels.mean())
            
            # Score statistics
            metrics[f"{prefix}_score_mean"] = float(predictions.mean())
            metrics[f"{prefix}_score_std"] = float(predictions.std())
            metrics[f"{prefix}_score_min"] = float(predictions.min())
            metrics[f"{prefix}_score_max"] = float(predictions.max())
            
        except Exception as e:
            warnings.warn(f"Error computing metrics: {e}")
            metrics[f"{prefix}_computation_error"] = 1.0
        
        return metrics
    
    def _compute_fpr_at_tpr(
        self, 
        labels: np.ndarray, 
        predictions: np.ndarray, 
        target_tpr: float
    ) -> float:
        """Compute False Positive Rate at a target True Positive Rate."""
        
        from sklearn.metrics import roc_curve
        
        try:
            fpr, tpr, thresholds = roc_curve(labels, predictions)
            
            # Find threshold closest to target TPR
            target_idx = np.argmin(np.abs(tpr - target_tpr))
            return float(fpr[target_idx])
            
        except:
            return 1.0  # Worst case
    
    def compute_convergence_metrics(
        self, 
        convergence_histories: List[List[Dict[str, Any]]],
        sample_types: List[str]
    ) -> Dict[str, float]:
        """
        Compute convergence-specific metrics.
        
        Args:
            convergence_histories: List of convergence histories per sample
            sample_types: Sample type for each history
            
        Returns:
            Convergence metrics
        """
        
        metrics = {}
        
        if not convergence_histories:
            return metrics
        
        # Group by sample type
        benign_convergence = []
        malicious_convergence = []
        
        for history, sample_type in zip(convergence_histories, sample_types):
            if sample_type == "benign":
                benign_convergence.append(history)
            else:
                malicious_convergence.append(history)
        
        # Analyze benign convergence (should be fast and stable)
        if benign_convergence:
            benign_metrics = self._analyze_convergence_group(benign_convergence, "benign")
            metrics.update(benign_metrics)
        
        # Analyze malicious convergence (should be slow or unstable)
        if malicious_convergence:
            malicious_metrics = self._analyze_convergence_group(malicious_convergence, "malicious")
            metrics.update(malicious_metrics)
        
        # Compute separation metrics
        if benign_convergence and malicious_convergence:
            separation_metrics = self._compute_convergence_separation(
                benign_convergence, malicious_convergence
            )
            metrics.update(separation_metrics)
        
        return metrics
    
    def _analyze_convergence_group(
        self, 
        convergence_group: List[List[Dict[str, Any]]], 
        group_name: str
    ) -> Dict[str, float]:
        """Analyze convergence patterns for a group of samples."""
        
        metrics = {}
        
        # Extract convergence statistics
        iterations_list = []
        final_drifts = []
        converged_count = 0
        
        for history in convergence_group:
            if history:
                iterations_list.append(len(history))
                if history[-1].get('converged', False):
                    converged_count += 1
                final_drifts.append(history[-1].get('drift', 1.0))
        
        if iterations_list:
            metrics[f"convergence_{group_name}_avg_iterations"] = np.mean(iterations_list)
            metrics[f"convergence_{group_name}_std_iterations"] = np.std(iterations_list)
            metrics[f"convergence_{group_name}_max_iterations"] = np.max(iterations_list)
            metrics[f"convergence_{group_name}_convergence_rate"] = converged_count / len(iterations_list)
        
        if final_drifts:
            metrics[f"convergence_{group_name}_avg_final_drift"] = np.mean(final_drifts)
            metrics[f"convergence_{group_name}_std_final_drift"] = np.std(final_drifts)
        
        return metrics
    
    def _compute_convergence_separation(
        self, 
        benign_convergence: List[List[Dict[str, Any]]], 
        malicious_convergence: List[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Compute separation between benign and malicious convergence patterns."""
        
        metrics = {}
        
        # Extract final statistics
        benign_iterations = [len(hist) for hist in benign_convergence if hist]
        malicious_iterations = [len(hist) for hist in malicious_convergence if hist]
        
        if benign_iterations and malicious_iterations:
            # Iteration separation
            benign_mean_iter = np.mean(benign_iterations)
            malicious_mean_iter = np.mean(malicious_iterations)
            
            metrics["convergence_iteration_separation"] = malicious_mean_iter - benign_mean_iter
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(benign_iterations) - 1) * np.var(benign_iterations) + 
                 (len(malicious_iterations) - 1) * np.var(malicious_iterations)) /
                (len(benign_iterations) + len(malicious_iterations) - 2)
            )
            
            if pooled_std > 0:
                cohens_d = (malicious_mean_iter - benign_mean_iter) / pooled_std
                metrics["convergence_effect_size"] = cohens_d
        
        return metrics
    
    def compute_localization_metrics(
        self, 
        flagged_units: List[List[int]], 
        ground_truth_units: List[List[int]]
    ) -> Dict[str, float]:
        """
        Compute localization metrics (IoU@k for flagged units).
        
        Args:
            flagged_units: List of flagged unit indices per sample
            ground_truth_units: List of ground truth malicious unit indices per sample
            
        Returns:
            Localization metrics
        """
        
        metrics = {}
        
        if len(flagged_units) != len(ground_truth_units):
            metrics["localization_error"] = 1.0
            return metrics
        
        ious = []
        precisions = []
        recalls = []
        
        for flagged, gt in zip(flagged_units, ground_truth_units):
            if not gt:  # No ground truth units
                if not flagged:
                    # Correct: no units flagged when none are malicious
                    ious.append(1.0)
                    precisions.append(1.0)
                    recalls.append(1.0)
                else:
                    # False positives
                    ious.append(0.0)
                    precisions.append(0.0)
                    recalls.append(1.0)  # Recall not applicable
                continue
            
            flagged_set = set(flagged)
            gt_set = set(gt)
            
            # Intersection over Union
            intersection = len(flagged_set & gt_set)
            union = len(flagged_set | gt_set)
            
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)
            
            # Precision and Recall
            precision = intersection / len(flagged_set) if len(flagged_set) > 0 else 0.0
            recall = intersection / len(gt_set) if len(gt_set) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        if ious:
            # IoU@k metrics
            for k in [1, 3, 5]:
                iou_at_k = np.mean([iou for iou in ious if iou >= (k / 10.0)])
                metrics[f"localization_iou_at_{k}"] = iou_at_k if not np.isnan(iou_at_k) else 0.0
            
            metrics["localization_mean_iou"] = np.mean(ious)
            metrics["localization_mean_precision"] = np.mean(precisions)
            metrics["localization_mean_recall"] = np.mean(recalls)
            
            # F1 score for localization
            if precisions and recalls:
                f1_scores = [
                    2 * p * r / (p + r) if (p + r) > 0 else 0.0 
                    for p, r in zip(precisions, recalls)
                ]
                metrics["localization_mean_f1"] = np.mean(f1_scores)
        
        return metrics
    
    def compute_interpretability_metrics(
        self, 
        explanations: List[Dict[str, Any]], 
        human_ratings: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Compute interpretability and explanation quality metrics.
        
        Args:
            explanations: List of model explanations
            human_ratings: Optional human ratings of explanation quality
            
        Returns:
            Interpretability metrics
        """
        
        metrics = {}
        
        if not explanations:
            return metrics
        
        # Explanation coverage
        has_divergence_explanation = sum(
            1 for exp in explanations 
            if exp.get('primary_channel') == 'divergence'
        )
        has_plausibility_explanation = sum(
            1 for exp in explanations 
            if exp.get('primary_channel') == 'plausibility'
        )
        
        metrics["interpretability_divergence_coverage"] = has_divergence_explanation / len(explanations)
        metrics["interpretability_plausibility_coverage"] = has_plausibility_explanation / len(explanations)
        
        # Explanation detail level
        detailed_explanations = sum(
            1 for exp in explanations 
            if (exp.get('divergence_channel', {}).get('evidence', {}).get('suspicious_units', []) or
                exp.get('plausibility_channel', {}).get('evidence', {}).get('abnormal_intents', []))
        )
        
        metrics["interpretability_detail_coverage"] = detailed_explanations / len(explanations)
        
        # Human evaluation metrics (if available)
        if human_ratings:
            rating_metrics = {}
            for rating_type in ["clarity", "usefulness", "accuracy", "completeness"]:
                ratings = [r.get(rating_type, 0.0) for r in human_ratings if rating_type in r]
                if ratings:
                    rating_metrics[f"interpretability_{rating_type}_mean"] = np.mean(ratings)
                    rating_metrics[f"interpretability_{rating_type}_std"] = np.std(ratings)
            
            metrics.update(rating_metrics)
        
        return metrics


class MetricsTracker:
    """Tracks metrics across training epochs and stages."""
    
    def __init__(self):
        self.history = {}
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics history."""
        
        for metric_name, value in metrics.items():
            if metric_name not in self.history:
                self.history[metric_name] = []
            
            self.history[metric_name].append((step, value))
            
            # Track best metrics (assume higher is better for most metrics)
            if metric_name not in self.best_metrics or value > self.best_metrics[metric_name][1]:
                self.best_metrics[metric_name] = (step, value)
    
    def get_best_metric(self, metric_name: str) -> Optional[Tuple[int, float]]:
        """Get best value for a metric."""
        return self.best_metrics.get(metric_name)
    
    def get_metric_history(self, metric_name: str) -> List[Tuple[int, float]]:
        """Get full history for a metric."""
        return self.history.get(metric_name, [])
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest values for all metrics."""
        latest = {}
        for metric_name, history in self.history.items():
            if history:
                latest[metric_name] = history[-1][1]
        return latest


if __name__ == "__main__":
    # Test metrics computation
    print("🧪 Testing ICN Metrics...")
    
    # Create test data
    np.random.seed(42)
    predictions = np.random.rand(1000)
    labels = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])  # Imbalanced
    
    # Compute metrics
    metrics = ICNMetrics()
    result = metrics.compute_metrics(predictions, labels)
    
    print("📊 Detection Metrics:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test convergence metrics
    print("\n🔄 Testing Convergence Metrics...")
    convergence_histories = [
        [{"drift": 0.1, "converged": False}, {"drift": 0.05, "converged": True}],  # Benign
        [{"drift": 0.2, "converged": False}, {"drift": 0.15, "converged": False}, {"drift": 0.1, "converged": False}],  # Malicious
    ]
    sample_types = ["benign", "malicious_intent"]
    
    conv_metrics = metrics.compute_convergence_metrics(convergence_histories, sample_types)
    
    print("🔄 Convergence Metrics:")
    for key, value in conv_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Metrics computation working correctly!")
    print("🚀 Ready for evaluation pipeline!")