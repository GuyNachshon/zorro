"""
Comprehensive evaluation system for CPG-GNN model.
"""

import logging
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CPGConfig, EvaluationConfig
from .model import CPGModel
from .graph_builder import CPGBuilder, CodePropertyGraph
from .trainer import PackageSample

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Results from comprehensive CPG-GNN evaluation."""
    
    # Classification metrics
    roc_auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fpr_at_95_tpr: float = 1.0
    
    # Speed benchmarks
    avg_inference_time: float = 0.0
    inference_times_by_size: Dict[str, float] = field(default_factory=dict)
    
    # Localization metrics
    localization_iou: float = 0.0
    attention_accuracy: float = 0.0
    
    # Robustness metrics
    obfuscation_drop: Dict[str, float] = field(default_factory=dict)
    avg_robustness_drop: float = 0.0
    
    # Interpretability metrics
    explanation_quality: float = 0.0
    api_prediction_accuracy: float = 0.0
    
    # Detailed results
    predictions: List[Dict] = field(default_factory=list)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""
    
    # Success criteria check
    meets_success_criteria: bool = False
    criteria_summary: Dict[str, bool] = field(default_factory=dict)


class CPGEvaluator:
    """Comprehensive evaluator for CPG-GNN model."""
    
    def __init__(self, 
                 model: CPGModel,
                 config: EvaluationConfig,
                 cpg_config: CPGConfig):
        self.model = model
        self.config = config
        self.cpg_config = cpg_config
        self.cpg_builder = CPGBuilder(cpg_config)
    
    def comprehensive_evaluation(self, 
                               test_samples: List[PackageSample],
                               synthetic_trojan_samples: Optional[List[PackageSample]] = None) -> EvaluationResults:
        """Run comprehensive evaluation of CPG-GNN model."""
        
        logger.info("Starting comprehensive CPG-GNN evaluation")
        
        results = EvaluationResults()
        
        # 1. Classification Performance
        logger.info("Evaluating classification performance...")
        classification_metrics = self._evaluate_classification(test_samples)
        results.roc_auc = classification_metrics['roc_auc']
        results.precision = classification_metrics['precision']
        results.recall = classification_metrics['recall']
        results.f1_score = classification_metrics['f1_score']
        results.fpr_at_95_tpr = classification_metrics['fpr_at_95_tpr']
        results.predictions = classification_metrics['predictions']
        results.confusion_matrix = classification_metrics['confusion_matrix']
        results.classification_report = classification_metrics['report']
        
        # 2. Speed Benchmarking
        logger.info("Running speed benchmarks...")
        speed_metrics = self._evaluate_speed(test_samples)
        results.avg_inference_time = speed_metrics['avg_time']
        results.inference_times_by_size = speed_metrics['times_by_size']
        
        # 3. Localization Analysis
        logger.info("Evaluating attention localization...")
        if synthetic_trojan_samples:
            localization_metrics = self._evaluate_localization(synthetic_trojan_samples)
            results.localization_iou = localization_metrics['iou']
            results.attention_accuracy = localization_metrics['accuracy']
        
        # 4. Robustness Testing
        logger.info("Testing robustness to obfuscation...")
        robustness_metrics = self._evaluate_robustness(test_samples)
        results.obfuscation_drop = robustness_metrics['drop_by_type']
        results.avg_robustness_drop = robustness_metrics['avg_drop']
        
        # 5. Interpretability Assessment
        logger.info("Assessing interpretability...")
        interpretability_metrics = self._evaluate_interpretability(test_samples)
        results.explanation_quality = interpretability_metrics['quality']
        results.api_prediction_accuracy = interpretability_metrics['api_accuracy']
        
        # 6. Check Success Criteria
        results.meets_success_criteria, results.criteria_summary = self._check_success_criteria(results)
        
        logger.info("Comprehensive evaluation completed")
        return results
    
    def _evaluate_classification(self, test_samples: List[PackageSample]) -> Dict[str, Any]:
        """Evaluate classification performance."""
        
        predictions = []
        probabilities = []
        labels = []
        
        self.model.eval()
        
        for sample in test_samples:
            # Build CPG
            cpg = self.cpg_builder.build_package_graph(
                sample.package_name,
                sample.ecosystem,
                sample.file_contents
            )
            
            # Predict
            with torch.no_grad():
                output = self.model.predict_package(cpg)
            
            predictions.append(output.prediction)
            probabilities.append(output.confidence if output.prediction == 1 else 1.0 - output.confidence)
            labels.append(sample.label)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        if len(set(labels)) > 1:
            roc_auc = roc_auc_score(labels, probabilities)
            
            # Calculate FPR at 95% TPR
            fpr_at_95_tpr = self._calculate_fpr_at_tpr(labels, probabilities, target_tpr=0.95)
        else:
            roc_auc = 0.5
            fpr_at_95_tpr = 1.0
        
        # Generate detailed results
        detailed_predictions = []
        for i, sample in enumerate(test_samples):
            detailed_predictions.append({
                "package_name": sample.package_name,
                "ecosystem": sample.ecosystem,
                "ground_truth": labels[i],
                "prediction": predictions[i],
                "probability": probabilities[i],
                "sample_type": sample.sample_type
            })
        
        cm = confusion_matrix(labels, predictions)
        report = classification_report(labels, predictions, target_names=['Benign', 'Malicious'])
        
        return {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fpr_at_95_tpr': fpr_at_95_tpr,
            'predictions': detailed_predictions,
            'confusion_matrix': cm,
            'report': report
        }
    
    def _evaluate_speed(self, test_samples: List[PackageSample]) -> Dict[str, Any]:
        """Evaluate inference speed across different package sizes."""
        
        # Categorize samples by size
        size_categories = {"small": [], "medium": [], "large": []}
        
        for sample in test_samples:
            num_files = len(sample.file_contents)
            if num_files < 10:
                size_categories["small"].append(sample)
            elif num_files < 50:
                size_categories["medium"].append(sample)
            else:
                size_categories["large"].append(sample)
        
        times_by_size = {}
        all_times = []
        
        self.model.eval()
        
        for size_category, samples in size_categories.items():
            if not samples:
                continue
                
            # Sample up to 20 packages per category
            test_samples_subset = random.sample(samples, min(20, len(samples)))
            
            category_times = []
            
            for sample in test_samples_subset:
                # Build CPG
                cpg = self.cpg_builder.build_package_graph(
                    sample.package_name,
                    sample.ecosystem,
                    sample.file_contents
                )
                
                # Time inference
                start_time = time.time()
                
                with torch.no_grad():
                    output = self.model.predict_package(cpg)
                
                inference_time = time.time() - start_time
                category_times.append(inference_time)
                all_times.append(inference_time)
            
            times_by_size[size_category] = np.mean(category_times) if category_times else 0.0
        
        return {
            'avg_time': np.mean(all_times) if all_times else 0.0,
            'times_by_size': times_by_size
        }
    
    def _evaluate_localization(self, synthetic_samples: List[PackageSample]) -> Dict[str, float]:
        """Evaluate attention-based localization on synthetic trojans."""
        
        if not synthetic_samples:
            return {'iou': 0.0, 'accuracy': 0.0}
        
        ious = []
        attention_accuracies = []
        
        self.model.eval()
        
        for sample in synthetic_samples:
            if 'injected_units' not in sample.metadata:
                continue
            
            # Build CPG
            cpg = self.cpg_builder.build_package_graph(
                sample.package_name,
                sample.ecosystem,
                sample.file_contents
            )
            
            # Get detailed prediction
            explanation = self.model.get_attention_explanation(cpg)
            
            if 'suspicious_subgraphs' not in explanation:
                continue
            
            # Get injected unit information
            injected_units = set(sample.metadata['injected_units'])
            
            # Get top-k suspicious units from attention
            suspicious_subgraphs = explanation['suspicious_subgraphs']
            if not suspicious_subgraphs:
                continue
                
            predicted_suspicious = set()
            for subgraph in suspicious_subgraphs[:self.config.top_k_subgraphs]:
                # Map back to file/unit names (simplified)
                predicted_suspicious.add(subgraph.get('original_node_id', ''))
            
            # Calculate IoU
            if injected_units or predicted_suspicious:
                intersection = len(injected_units & predicted_suspicious)
                union = len(injected_units | predicted_suspicious)
                iou = intersection / union if union > 0 else 0.0
                ious.append(iou)
            
            # Calculate attention accuracy (top-1)
            if suspicious_subgraphs and injected_units:
                top_prediction = suspicious_subgraphs[0].get('original_node_id', '')
                accuracy = 1.0 if top_prediction in injected_units else 0.0
                attention_accuracies.append(accuracy)
        
        return {
            'iou': np.mean(ious) if ious else 0.0,
            'accuracy': np.mean(attention_accuracies) if attention_accuracies else 0.0
        }
    
    def _evaluate_robustness(self, test_samples: List[PackageSample]) -> Dict[str, Any]:
        """Evaluate robustness to different obfuscation types."""
        
        obfuscation_types = self.config.test_obfuscation_types
        drop_by_type = {}
        
        # Get baseline performance
        baseline_predictions = self._get_baseline_predictions(test_samples)
        
        for obfuscation_type in obfuscation_types:
            logger.info(f"Testing robustness to {obfuscation_type}")
            
            # Apply obfuscation
            obfuscated_samples = self._apply_obfuscation(test_samples, obfuscation_type)
            
            # Get obfuscated predictions
            obfuscated_predictions = self._get_baseline_predictions(obfuscated_samples)
            
            # Calculate performance drop
            baseline_auc = self._calculate_auc(baseline_predictions)
            obfuscated_auc = self._calculate_auc(obfuscated_predictions)
            
            drop = max(0.0, baseline_auc - obfuscated_auc)
            drop_by_type[obfuscation_type] = drop
        
        avg_drop = np.mean(list(drop_by_type.values())) if drop_by_type else 0.0
        
        return {
            'drop_by_type': drop_by_type,
            'avg_drop': avg_drop
        }
    
    def _evaluate_interpretability(self, test_samples: List[PackageSample]) -> Dict[str, float]:
        """Evaluate interpretability and explanation quality."""
        
        explanation_scores = []
        api_accuracies = []
        
        self.model.eval()
        
        # Sample subset for detailed analysis
        sample_subset = random.sample(test_samples, min(50, len(test_samples)))
        
        for sample in sample_subset:
            # Build CPG
            cpg = self.cpg_builder.build_package_graph(
                sample.package_name,
                sample.ecosystem,
                sample.file_contents
            )
            
            # Get explanation
            explanation = self.model.get_attention_explanation(cpg)
            
            # Score explanation quality
            explanation_score = self._score_explanation(explanation, sample)
            explanation_scores.append(explanation_score)
            
            # Check API prediction accuracy
            if 'api_analysis' in explanation:
                api_accuracy = self._check_api_accuracy(explanation['api_analysis'], cpg)
                api_accuracies.append(api_accuracy)
        
        return {
            'quality': np.mean(explanation_scores) if explanation_scores else 0.0,
            'api_accuracy': np.mean(api_accuracies) if api_accuracies else 0.0
        }
    
    def _check_success_criteria(self, results: EvaluationResults) -> Tuple[bool, Dict[str, bool]]:
        """Check if model meets success criteria."""
        
        criteria = {
            'detection_auc': results.roc_auc >= self.config.target_roc_auc,
            'false_positive_rate': results.fpr_at_95_tpr <= self.config.target_fpr,
            'localization_iou': results.localization_iou >= self.config.target_localization_iou,
            'inference_speed': results.avg_inference_time <= self.config.max_inference_time_seconds,
            'robustness': results.avg_robustness_drop <= self.config.obfuscation_drop_threshold
        }
        
        meets_all = all(criteria.values())
        
        return meets_all, criteria
    
    def _get_baseline_predictions(self, samples: List[PackageSample]) -> List[Dict]:
        """Get predictions for baseline performance calculation."""
        
        predictions = []
        
        self.model.eval()
        
        for sample in samples:
            cpg = self.cpg_builder.build_package_graph(
                sample.package_name,
                sample.ecosystem,
                sample.file_contents
            )
            
            with torch.no_grad():
                output = self.model.predict_package(cpg)
            
            predictions.append({
                'label': sample.label,
                'prediction': output.prediction,
                'probability': output.confidence if output.prediction == 1 else 1.0 - output.confidence
            })
        
        return predictions
    
    def _apply_obfuscation(self, 
                          samples: List[PackageSample], 
                          obfuscation_type: str) -> List[PackageSample]:
        """Apply obfuscation to test samples."""
        
        obfuscated_samples = []
        
        for sample in samples:
            obfuscated_contents = {}
            
            for file_path, content in sample.file_contents.items():
                if obfuscation_type == "minification":
                    obfuscated_content = self._minify_code(content)
                elif obfuscation_type == "base64_encoding":
                    obfuscated_content = self._encode_strings_base64(content)
                elif obfuscation_type == "variable_renaming":
                    obfuscated_content = self._rename_variables(content)
                elif obfuscation_type == "dead_code_injection":
                    obfuscated_content = self._inject_dead_code(content)
                else:
                    obfuscated_content = content
                
                obfuscated_contents[file_path] = obfuscated_content
            
            obfuscated_sample = PackageSample(
                package_name=sample.package_name,
                ecosystem=sample.ecosystem,
                label=sample.label,
                file_contents=obfuscated_contents,
                sample_type=sample.sample_type,
                metadata=sample.metadata.copy()
            )
            
            obfuscated_samples.append(obfuscated_sample)
        
        return obfuscated_samples
    
    def _calculate_auc(self, predictions: List[Dict]) -> float:
        """Calculate AUC from predictions."""
        
        labels = [p['label'] for p in predictions]
        probs = [p['probability'] for p in predictions]
        
        if len(set(labels)) > 1:
            return roc_auc_score(labels, probs)
        else:
            return 0.5
    
    def _calculate_fpr_at_tpr(self, labels: List[int], probabilities: List[float], target_tpr: float = 0.95) -> float:
        """Calculate false positive rate at target true positive rate."""
        
        from sklearn.metrics import roc_curve
        
        if len(set(labels)) <= 1:
            return 1.0
        
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        
        # Find FPR at target TPR
        tpr_idx = np.where(tpr >= target_tpr)[0]
        if len(tpr_idx) > 0:
            return fpr[tpr_idx[0]]
        else:
            return 1.0
    
    def _score_explanation(self, explanation: Dict, sample: PackageSample) -> float:
        """Score quality of explanation."""
        
        score = 0.0
        max_score = 4.0
        
        # Check if prediction makes sense
        if explanation.get('prediction', {}).get('is_malicious') == (sample.label == 1):
            score += 1.0
        
        # Check if suspicious subgraphs are provided for malicious samples
        if sample.label == 1 and explanation.get('suspicious_subgraphs'):
            score += 1.0
        
        # Check if API analysis is meaningful
        if explanation.get('api_analysis', {}).get('predicted_risky_apis'):
            score += 1.0
        
        # Check attention analysis
        if explanation.get('attention_analysis', {}).get('attention_entropy', 0) > 0:
            score += 1.0
        
        return score / max_score
    
    def _check_api_accuracy(self, api_analysis: Dict, cpg: CodePropertyGraph) -> float:
        """Check accuracy of API predictions."""
        
        predicted_apis = api_analysis.get('predicted_risky_apis', [])
        actual_apis = cpg.api_calls
        
        if not predicted_apis:
            return 0.0
        
        # Check if top predicted APIs are actually present
        correct_predictions = 0
        for api_info in predicted_apis[:3]:  # Top 3
            api_name = api_info.get('api', '')
            if api_name in actual_apis:
                correct_predictions += 1
        
        return correct_predictions / min(3, len(predicted_apis))
    
    def _minify_code(self, content: str) -> str:
        """Minify code by removing whitespace."""
        lines = content.split('\n')
        minified_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(minified_lines)
    
    def _encode_strings_base64(self, content: str) -> str:
        """Encode string literals with base64."""
        import re
        import base64
        
        def encode_string(match):
            string_content = match.group(1)
            if len(string_content) > 3:
                encoded = base64.b64encode(string_content.encode()).decode()
                return f'base64.b64decode("{encoded}").decode()'
            return match.group(0)
        
        return re.sub(r'["\']([^"\']+)["\']', encode_string, content)
    
    def _rename_variables(self, content: str) -> str:
        """Rename variables with obfuscated names."""
        import re
        
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        variables = set(re.findall(var_pattern, content))
        
        for var in list(variables)[:5]:  # Limit to avoid breaking code
            if len(var) > 3:
                obfuscated = f"_0x{abs(hash(var)) % 10000:04x}"
                content = content.replace(var, obfuscated)
        
        return content
    
    def _inject_dead_code(self, content: str) -> str:
        """Inject dead code that doesn't affect functionality."""
        dead_code = "\n# Dead code injection\nif False:\n    unused_var = 42\n"
        return content + dead_code
    
    def generate_evaluation_report(self, results: EvaluationResults) -> str:
        """Generate comprehensive evaluation report."""
        
        report = ["=" * 60]
        report.append("CPG-GNN COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 60)
        
        # Success criteria summary
        report.append(f"\n🎯 SUCCESS CRITERIA: {'✅ PASSED' if results.meets_success_criteria else '❌ FAILED'}")
        report.append("-" * 40)
        
        for criterion, passed in results.criteria_summary.items():
            status = "✅" if passed else "❌"
            report.append(f"{status} {criterion}: {passed}")
        
        # Classification performance
        report.append(f"\n📊 CLASSIFICATION PERFORMANCE")
        report.append("-" * 40)
        report.append(f"ROC-AUC: {results.roc_auc:.4f} (target: ≥{self.config.target_roc_auc:.2f})")
        report.append(f"Precision: {results.precision:.4f}")
        report.append(f"Recall: {results.recall:.4f}")
        report.append(f"F1-Score: {results.f1_score:.4f}")
        report.append(f"FPR@95%TPR: {results.fpr_at_95_tpr:.4f} (target: ≤{self.config.target_fpr:.2f})")
        
        # Speed performance
        report.append(f"\n⚡ SPEED PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Average inference time: {results.avg_inference_time:.2f}s (target: ≤{self.config.max_inference_time_seconds:.0f}s)")
        
        for size, time_val in results.inference_times_by_size.items():
            report.append(f"  {size} packages: {time_val:.2f}s")
        
        # Localization performance
        report.append(f"\n🎯 LOCALIZATION PERFORMANCE")
        report.append("-" * 40)
        report.append(f"IoU Score: {results.localization_iou:.4f} (target: ≥{self.config.target_localization_iou:.2f})")
        report.append(f"Attention Accuracy: {results.attention_accuracy:.4f}")
        
        # Robustness
        report.append(f"\n🛡️ ROBUSTNESS TO OBFUSCATION")
        report.append("-" * 40)
        report.append(f"Average performance drop: {results.avg_robustness_drop:.4f} (target: ≤{self.config.obfuscation_drop_threshold:.2f})")
        
        for obfuscation_type, drop in results.obfuscation_drop.items():
            report.append(f"  {obfuscation_type}: {drop:.4f} drop")
        
        # Interpretability
        report.append(f"\n🔍 INTERPRETABILITY")
        report.append("-" * 40)
        report.append(f"Explanation Quality: {results.explanation_quality:.4f}")
        report.append(f"API Prediction Accuracy: {results.api_prediction_accuracy:.4f}")
        
        # Classification report
        report.append(f"\n📋 DETAILED CLASSIFICATION REPORT")
        report.append("-" * 40)
        report.append(results.classification_report)
        
        return "\n".join(report)