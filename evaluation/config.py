"""
Configuration management for evaluation system.
Supports YAML-based configuration with validation.
"""

import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    type: str  # openrouter, huggingface, icn, amil, cpg, neobert, baseline
    enabled: bool = True

    # OpenRouter specific
    openrouter_id: Optional[str] = None

    # HuggingFace specific
    hf_model_id: Optional[str] = None

    # Local model specific
    model_path: Optional[str] = None

    # Additional parameters
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    """Configuration for prompt strategies."""
    zero_shot: bool = True
    few_shot: bool = True
    reasoning: bool = True
    file_by_file: bool = False

    # Custom prompts
    custom_prompts: Dict[str, str] = field(default_factory=dict)

    # Few-shot examples
    few_shot_examples_count: int = 3

    # Token limits
    max_tokens: Dict[str, int] = field(default_factory=lambda: {
        "zero_shot": 1000,
        "few_shot": 1500,
        "reasoning": 2000,
        "file_by_file": 800
    })


@dataclass
class DataConfig:
    """Configuration for benchmark data."""
    max_samples_per_category: int = 50
    test_split_ratio: float = 0.3
    use_cached_data: bool = True
    data_path: Optional[str] = None

    # Sample categories to include
    include_categories: List[str] = field(default_factory=lambda: [
        "benign", "compromised_lib", "malicious_intent"
    ])


@dataclass
class ExecutionConfig:
    """Configuration for benchmark execution."""
    max_concurrent_requests: int = 2
    cost_limit_usd: float = 25.0
    timeout_seconds: int = 120
    retry_failed: bool = True
    max_retries: int = 3

    # Rate limiting
    delay_between_requests: float = 0.1
    requests_per_minute: int = 60


@dataclass
class OutputConfig:
    """Configuration for output and reporting."""
    output_directory: str = "evaluation_results"
    save_raw_results: bool = True
    save_analysis: bool = True
    save_visualizations: bool = True

    # Report formats
    generate_markdown_report: bool = True
    generate_json_summary: bool = True
    generate_csv_export: bool = False

    # Visualization options
    create_heatmaps: bool = True
    create_scatter_plots: bool = True
    create_ranking_charts: bool = True


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration."""
    # Basic info
    name: str = "default_evaluation"
    description: str = ""

    # Component configs
    models: List[ModelConfig] = field(default_factory=list)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # API keys and credentials
    api_keys: Dict[str, str] = field(default_factory=dict)

    # Advanced settings
    statistical_significance_level: float = 0.05
    enable_statistical_testing: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """Create config from dictionary."""
        # Parse models
        models = []
        for model_data in data.get('models', []):
            if isinstance(model_data, dict):
                models.append(ModelConfig(**model_data))
            else:
                # Simple string format
                models.append(ModelConfig(name=model_data, type="openrouter"))

        # Parse other components
        prompts = PromptConfig(**data.get('prompts', {}))
        data_config = DataConfig(**data.get('data', {}))
        execution = ExecutionConfig(**data.get('execution', {}))
        output = OutputConfig(**data.get('output', {}))

        # Get API keys from environment if not in config
        api_keys = data.get('api_keys', {})
        if 'openrouter' not in api_keys:
            api_keys['openrouter'] = os.getenv('OPENROUTER_API_KEY')

        return cls(
            name=data.get('name', 'default_evaluation'),
            description=data.get('description', ''),
            models=models,
            prompts=prompts,
            data=data_config,
            execution=execution,
            output=output,
            api_keys=api_keys,
            statistical_significance_level=data.get('statistical_significance_level', 0.05),
            enable_statistical_testing=data.get('enable_statistical_testing', True)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'models': [
                {
                    'name': m.name,
                    'type': m.type,
                    'enabled': m.enabled,
                    'openrouter_id': m.openrouter_id,
                    'hf_model_id': m.hf_model_id,
                    'model_path': m.model_path,
                    'parameters': m.parameters
                } for m in self.models
            ],
            'prompts': {
                'zero_shot': self.prompts.zero_shot,
                'few_shot': self.prompts.few_shot,
                'reasoning': self.prompts.reasoning,
                'file_by_file': self.prompts.file_by_file,
                'custom_prompts': self.prompts.custom_prompts,
                'few_shot_examples_count': self.prompts.few_shot_examples_count,
                'max_tokens': self.prompts.max_tokens
            },
            'data': {
                'max_samples_per_category': self.data.max_samples_per_category,
                'test_split_ratio': self.data.test_split_ratio,
                'use_cached_data': self.data.use_cached_data,
                'data_path': self.data.data_path,
                'include_categories': self.data.include_categories
            },
            'execution': {
                'max_concurrent_requests': self.execution.max_concurrent_requests,
                'cost_limit_usd': self.execution.cost_limit_usd,
                'timeout_seconds': self.execution.timeout_seconds,
                'retry_failed': self.execution.retry_failed,
                'max_retries': self.execution.max_retries,
                'delay_between_requests': self.execution.delay_between_requests,
                'requests_per_minute': self.execution.requests_per_minute
            },
            'output': {
                'output_directory': self.output.output_directory,
                'save_raw_results': self.output.save_raw_results,
                'save_analysis': self.output.save_analysis,
                'save_visualizations': self.output.save_visualizations,
                'generate_markdown_report': self.output.generate_markdown_report,
                'generate_json_summary': self.output.generate_json_summary,
                'generate_csv_export': self.output.generate_csv_export,
                'create_heatmaps': self.output.create_heatmaps,
                'create_scatter_plots': self.output.create_scatter_plots,
                'create_ranking_charts': self.output.create_ranking_charts
            },
            'api_keys': self.api_keys,
            'statistical_significance_level': self.statistical_significance_level,
            'enable_statistical_testing': self.enable_statistical_testing
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check if we have any models
        enabled_models = [m for m in self.models if m.enabled]
        if not enabled_models:
            issues.append("No enabled models configured")

        # Check OpenRouter models have API key
        openrouter_models = [m for m in enabled_models if m.type == "openrouter"]
        if openrouter_models and not self.api_keys.get('openrouter'):
            issues.append("OpenRouter models configured but no API key provided")

        # Check local models have paths (or will be trained)
        local_models = [m for m in enabled_models if m.type in ["icn", "amil", "cpg", "neobert"]]
        for model in local_models:
            if not model.model_path:
                # It's OK if no path is provided - models can be trained on demand
                pass

        # Check data configuration
        if self.data.max_samples_per_category <= 0:
            issues.append("max_samples_per_category must be positive")

        if not (0 < self.data.test_split_ratio < 1):
            issues.append("test_split_ratio must be between 0 and 1")

        # Check execution configuration
        if self.execution.max_concurrent_requests <= 0:
            issues.append("max_concurrent_requests must be positive")

        if self.execution.cost_limit_usd <= 0:
            issues.append("cost_limit_usd must be positive")

        return issues


def load_config(config_path: Union[str, Path]) -> EvaluationConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        config = EvaluationConfig.from_dict(data)

        # Validate configuration
        issues = config.validate()
        if issues:
            logger.warning(f"Configuration validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")

        logger.info(f"✅ Loaded configuration: {config.name}")
        logger.info(f"   Models: {len([m for m in config.models if m.enabled])}")
        logger.info(f"   Output: {config.output.output_directory}")

        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def save_config(config: EvaluationConfig, config_path: Union[str, Path]):
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)

        logger.info(f"✅ Configuration saved to {config_path}")

    except Exception as e:
        raise ValueError(f"Error saving configuration: {e}")


def create_example_configs():
    """Create example configuration files."""
    configs_dir = Path("evaluation/configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Quick test config
    quick_config = EvaluationConfig(
        name="quick_test",
        description="Quick test with 2 models, basic prompts, minimal samples",
        models=[
            ModelConfig(name="gpt4o", type="openrouter", openrouter_id="openai/gpt-4o"),
            ModelConfig(name="claude", type="openrouter", openrouter_id="anthropic/claude-3.5-sonnet")
        ],
        prompts=PromptConfig(
            zero_shot=True,
            few_shot=False,
            reasoning=False,
            file_by_file=False
        ),
        data=DataConfig(max_samples_per_category=10),
        execution=ExecutionConfig(cost_limit_usd=5.0, max_concurrent_requests=1),
        output=OutputConfig(output_directory="quick_test_results")
    )

    # External models comprehensive config
    external_config = EvaluationConfig(
        name="external_models_comprehensive",
        description="Comprehensive evaluation of all external models with multiple prompts",
        models=[
            ModelConfig(name="gpt4o", type="openrouter", openrouter_id="openai/gpt-4o"),
            ModelConfig(name="o1_preview", type="openrouter", openrouter_id="openai/o1-preview"),
            ModelConfig(name="claude", type="openrouter", openrouter_id="anthropic/claude-3.5-sonnet"),
            ModelConfig(name="gemini_pro", type="openrouter", openrouter_id="google/gemini-pro-1.5"),
            ModelConfig(name="qwen_coder", type="openrouter", openrouter_id="qwen/qwen-3-coder-32b-instruct"),
            ModelConfig(name="deepseek", type="openrouter", openrouter_id="deepseek/deepseek-coder-v2-instruct"),
            ModelConfig(name="llama33", type="openrouter", openrouter_id="meta-llama/llama-3.3-70b-instruct")
        ],
        prompts=PromptConfig(
            zero_shot=True,
            few_shot=True,
            reasoning=True,
            file_by_file=False  # Expensive, disabled by default
        ),
        data=DataConfig(max_samples_per_category=50),
        execution=ExecutionConfig(cost_limit_usd=40.0, max_concurrent_requests=2),
        output=OutputConfig(
            output_directory="external_comprehensive_results",
            save_visualizations=True,
            create_heatmaps=True,
            create_scatter_plots=True
        )
    )

    # Local models only config
    local_config = EvaluationConfig(
        name="local_models_only",
        description="Evaluation of local ICN/AMIL models vs baselines",
        models=[
            ModelConfig(name="icn", type="icn", model_path="checkpoints/icn_model.pth"),
            ModelConfig(name="amil", type="amil", model_path="checkpoints/amil_model.pth"),
            ModelConfig(name="heuristic_baseline", type="baseline", parameters={"baseline_type": "heuristic"}),
            ModelConfig(name="random_baseline", type="baseline", parameters={"baseline_type": "random"})
        ],
        prompts=PromptConfig(
            zero_shot=False,
            few_shot=False,
            reasoning=False,
            file_by_file=False
        ),  # Local models don't use prompts
        data=DataConfig(max_samples_per_category=100),
        execution=ExecutionConfig(cost_limit_usd=0.0, max_concurrent_requests=4),
        output=OutputConfig(output_directory="local_models_results")
    )

    # Save example configs
    save_config(quick_config, configs_dir / "quick_test.yaml")
    save_config(external_config, configs_dir / "external_comprehensive.yaml")
    save_config(local_config, configs_dir / "local_models.yaml")

    logger.info(f"✅ Created example configurations in {configs_dir}/")


if __name__ == "__main__":
    # Create example configurations
    create_example_configs()