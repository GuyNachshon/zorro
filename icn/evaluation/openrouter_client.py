"""
OpenRouter API client for benchmarking multiple LLMs.
Provides unified interface to GPT, Claude, Gemini, and open source models.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import aiohttp
import os
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an OpenRouter model."""
    name: str
    provider: str
    context_length: int
    cost_per_1k_tokens: float
    supports_reasoning: bool = False
    description: str = ""


@dataclass
class BenchmarkRequest:
    """Request for model evaluation."""
    prompt: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResponse:
    """Response from model evaluation."""
    model_name: str
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_seconds: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    """Client for OpenRouter API with comprehensive model support."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY environment variable.")
        
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Model configurations for benchmarking
        self.models = self._initialize_model_configs()
        
        # Usage tracking
        self.total_cost = 0.0
        self.total_requests = 0
        self.request_log: List[BenchmarkResponse] = []
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize configurations for available models."""
        return {
            # OpenAI Models
            "openai/gpt-4o": ModelConfig(
                name="openai/gpt-4o",
                provider="OpenAI",
                context_length=128000,
                cost_per_1k_tokens=0.015,  # Approximate
                description="Latest GPT-4o model"
            ),
            "openai/o1-preview": ModelConfig(
                name="openai/o1-preview", 
                provider="OpenAI",
                context_length=128000,
                cost_per_1k_tokens=0.060,  # Higher cost for reasoning
                supports_reasoning=True,
                description="GPT o1 reasoning model"
            ),
            "openai/o1-mini": ModelConfig(
                name="openai/o1-mini",
                provider="OpenAI", 
                context_length=128000,
                cost_per_1k_tokens=0.012,
                supports_reasoning=True,
                description="GPT o1 mini reasoning model"
            ),
            
            # Anthropic Models
            "anthropic/claude-3.5-sonnet": ModelConfig(
                name="anthropic/claude-3.5-sonnet",
                provider="Anthropic",
                context_length=200000,
                cost_per_1k_tokens=0.015,
                description="Latest Claude 3.5 Sonnet"
            ),
            
            # Google Models
            "google/gemini-pro-1.5": ModelConfig(
                name="google/gemini-pro-1.5",
                provider="Google",
                context_length=1000000,
                cost_per_1k_tokens=0.007,
                description="Gemini 1.5 Pro"
            ),
            "google/gemini-flash-2.0": ModelConfig(
                name="google/gemini-flash-2.0", 
                provider="Google",
                context_length=1000000,
                cost_per_1k_tokens=0.002,
                description="Gemini 2.0 Flash"
            ),
            
            # Open Source Models
            "qwen/qwen-3-coder-32b-instruct": ModelConfig(
                name="qwen/qwen-3-coder-32b-instruct",
                provider="Qwen",
                context_length=32000,
                cost_per_1k_tokens=0.001,
                description="Qwen 3 Coder model"
            ),
            "nousresearch/hermes-3-llama-3.1-405b": ModelConfig(
                name="nousresearch/hermes-3-llama-3.1-405b",
                provider="Nous Research",
                context_length=128000, 
                cost_per_1k_tokens=0.005,
                description="Hermes 3 based on Llama 3.1"
            ),
            "deepseek/deepseek-coder-v2-instruct": ModelConfig(
                name="deepseek/deepseek-coder-v2-instruct",
                provider="DeepSeek",
                context_length=128000,
                cost_per_1k_tokens=0.002,
                description="DeepSeek Coder V2"
            ),
            "meta-llama/llama-3.3-70b-instruct": ModelConfig(
                name="meta-llama/llama-3.3-70b-instruct", 
                provider="Meta",
                context_length=128000,
                cost_per_1k_tokens=0.003,
                description="Latest Llama 3.3"
            ),
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),  # 2 minute timeout
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/baddon-ai/zorro",  # Required by OpenRouter
                "X-Title": "ICN Malware Detection Benchmark"  # Optional but helpful
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def generate_response(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """Generate response from specified model."""
        
        if not self.session:
            raise RuntimeError("Client must be used within async context manager")
        
        start_time = time.time()
        model_config = self.models.get(request.model_name)
        
        if not model_config:
            return BenchmarkResponse(
                model_name=request.model_name,
                response_text="",
                prompt_tokens=0,
                completion_tokens=0, 
                total_tokens=0,
                cost_usd=0.0,
                latency_seconds=0.0,
                success=False,
                error_message=f"Model {request.model_name} not configured",
                metadata=request.metadata
            )
        
        # Prepare API request
        api_request = {
            "model": request.model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": request.prompt
                }
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        
        try:
            # Make API request
            async with self.session.post(f"{self.base_url}/chat/completions", json=api_request) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    error_msg = response_data.get("error", {}).get("message", f"HTTP {response.status}")
                    return BenchmarkResponse(
                        model_name=request.model_name,
                        response_text="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0, 
                        cost_usd=0.0,
                        latency_seconds=time.time() - start_time,
                        success=False,
                        error_message=error_msg,
                        metadata=request.metadata
                    )
                
                # Parse successful response
                choice = response_data["choices"][0]
                usage = response_data.get("usage", {})
                
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                
                # Calculate cost
                cost_usd = (total_tokens / 1000) * model_config.cost_per_1k_tokens
                
                response_obj = BenchmarkResponse(
                    model_name=request.model_name,
                    response_text=choice["message"]["content"].strip(),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    latency_seconds=time.time() - start_time,
                    success=True,
                    metadata=request.metadata
                )
                
                # Update tracking
                self.total_cost += cost_usd
                self.total_requests += 1
                self.request_log.append(response_obj)
                
                return response_obj
                
        except asyncio.TimeoutError:
            return BenchmarkResponse(
                model_name=request.model_name,
                response_text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_seconds=time.time() - start_time,
                success=False,
                error_message="Request timeout",
                metadata=request.metadata
            )
            
        except Exception as e:
            return BenchmarkResponse(
                model_name=request.model_name,
                response_text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_seconds=time.time() - start_time,
                success=False,
                error_message=str(e),
                metadata=request.metadata
            )
    
    async def batch_generate(
        self, 
        requests: List[BenchmarkRequest],
        max_concurrent: int = 5,
        delay_between_requests: float = 0.1
    ) -> List[BenchmarkResponse]:
        """Generate responses for multiple requests with concurrency control."""
        
        async def process_request_with_delay(request: BenchmarkRequest, delay: float) -> BenchmarkResponse:
            await asyncio.sleep(delay)
            return await self.generate_response(request)
        
        # Create tasks with staggered delays to respect rate limits
        tasks = []
        for i, request in enumerate(requests):
            delay = i * delay_between_requests
            task = process_request_with_delay(request, delay)
            tasks.append(task)
        
        # Run with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_task(task) for task in tasks]
        responses = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Create error response
                error_response = BenchmarkResponse(
                    model_name=requests[i].model_name,
                    response_text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    latency_seconds=0.0,
                    success=False,
                    error_message=str(response),
                    metadata=requests[i].metadata
                )
                results.append(error_response)
            else:
                results.append(response)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[ModelConfig]:
        """Get information about a specific model."""
        return self.models.get(model_name)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of API usage and costs."""
        successful_requests = sum(1 for log in self.request_log if log.success)
        failed_requests = self.total_requests - successful_requests
        
        total_tokens = sum(log.total_tokens for log in self.request_log)
        avg_latency = sum(log.latency_seconds for log in self.request_log) / len(self.request_log) if self.request_log else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_cost_usd": self.total_cost,
            "total_tokens": total_tokens,
            "average_latency_seconds": avg_latency,
            "cost_per_request": self.total_cost / self.total_requests if self.total_requests > 0 else 0
        }
    
    def save_usage_log(self, filepath: Path):
        """Save detailed usage log to file."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_usage_summary(),
            "detailed_log": [
                {
                    "model_name": log.model_name,
                    "success": log.success,
                    "prompt_tokens": log.prompt_tokens,
                    "completion_tokens": log.completion_tokens, 
                    "total_tokens": log.total_tokens,
                    "cost_usd": log.cost_usd,
                    "latency_seconds": log.latency_seconds,
                    "error_message": log.error_message,
                    "metadata": log.metadata
                }
                for log in self.request_log
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)


# Utility functions for common prompts
class MaliciousPackagePrompts:
    """Standard prompts for malicious package detection."""
    
    @staticmethod
    def zero_shot_prompt(package_content: str) -> str:
        """Zero-shot malicious package detection prompt."""
        return f"""Analyze the following package code for malicious behavior. Respond with a JSON object containing:
- "is_malicious": boolean (true if malicious, false if benign)  
- "confidence": float (0.0 to 1.0)
- "reasoning": string (explanation of your analysis)
- "malicious_indicators": list of strings (specific indicators found)

Package code:
```
{package_content}
```

Response (JSON only):"""
    
    @staticmethod 
    def few_shot_prompt(package_content: str, examples: List[Dict[str, Any]]) -> str:
        """Few-shot prompt with examples."""
        
        examples_text = ""
        for i, example in enumerate(examples, 1):
            label = "malicious" if example["is_malicious"] else "benign"
            examples_text += f"""
Example {i} ({label}):
```
{example["content"]}
```
Analysis: {example["reasoning"]}

"""
        
        return f"""Analyze packages for malicious behavior. Here are some examples:

{examples_text}

Now analyze this package:
```
{package_content}
```

Respond with a JSON object containing:
- "is_malicious": boolean
- "confidence": float (0.0 to 1.0) 
- "reasoning": string
- "malicious_indicators": list of strings

Response (JSON only):"""
    
    @staticmethod
    def reasoning_prompt(package_content: str) -> str:
        """Prompt designed for reasoning models like GPT o1."""
        return f"""<thinking>
I need to carefully analyze this package code for malicious behavior. Let me think through this step by step:

1. First, I'll examine the package structure and metadata
2. Then I'll look for suspicious API calls or patterns
3. I'll check for obfuscation or encoding techniques
4. I'll assess the installation scripts and runtime behavior
5. Finally, I'll make a determination based on the evidence

Let me analyze this systematically...
</thinking>

Analyze this package code for malicious behavior:

```
{package_content}
```

Provide a thorough analysis including:
- Overall assessment (malicious/benign)
- Confidence level (0.0 to 1.0)
- Step-by-step reasoning
- Specific malicious indicators (if any)
- Risk assessment

Response as JSON:
{{"is_malicious": boolean, "confidence": float, "reasoning": "detailed explanation", "malicious_indicators": [list]}}"""
    
    @staticmethod
    def file_by_file_prompt(file_path: str, file_content: str) -> str:
        """Prompt for analyzing individual files."""
        return f"""Analyze this individual file from a package for malicious behavior:

File: {file_path}

```
{file_content}
```

Focus on:
- Suspicious code patterns in this specific file
- Potentially dangerous function calls
- Obfuscation or encoding techniques
- Network operations or file system access
- Any signs of malicious intent

Respond with JSON containing:
- "is_malicious": boolean (true if this file is malicious)
- "confidence": float (0.0 to 1.0)  
- "reasoning": string (explanation of analysis)
- "malicious_indicators": list of strings (specific suspicious patterns found)
- "risk_level": string ("high", "medium", "low", "none")

Response (JSON only):"""
    
    @staticmethod 
    def package_aggregation_prompt(package_name: str, file_analyses: List[Dict]) -> str:
        """Prompt for aggregating individual file analyses into package assessment."""
        
        # Create summary of individual file results
        file_summaries = []
        for analysis in file_analyses:
            file_path = analysis.get('file_path', 'unknown')
            is_malicious = analysis.get('is_malicious', False)
            confidence = analysis.get('confidence', 0.0)
            risk_level = analysis.get('risk_level', 'none')
            reasoning = analysis.get('reasoning', 'No analysis')[:100] + "..."
            
            file_summaries.append(f"- {file_path}: {'MALICIOUS' if is_malicious else 'BENIGN'} "
                                f"(confidence: {confidence:.2f}, risk: {risk_level}) - {reasoning}")
        
        summaries_text = "\n".join(file_summaries)
        
        return f"""Based on individual file analyses, make an overall assessment of this package:

Package: {package_name}
Individual File Analysis Results:

{summaries_text}

Consider:
- How many files appear malicious vs benign
- Severity of threats found in individual files
- Whether malicious files are core functionality or auxiliary
- Overall risk to systems if this package were installed

Provide a comprehensive package-level assessment:

Response as JSON:
{{
    "is_malicious": boolean,
    "confidence": float,
    "reasoning": "detailed explanation of package-level assessment", 
    "malicious_indicators": ["list", "of", "package-level", "threats"],
    "aggregation_method": "description of how you combined file analyses",
    "risk_assessment": "overall risk evaluation"
}}"""


if __name__ == "__main__":
    # Test the OpenRouter client
    import asyncio
    import os
    
    async def test_client():
        # Test with a simple request
        if not os.getenv("OPENROUTER_API_KEY"):
            print("❌ Set OPENROUTER_API_KEY environment variable to test")
            return
        
        async with OpenRouterClient() as client:
            print(f"🔧 Available models: {len(client.get_available_models())}")
            
            # Test request
            request = BenchmarkRequest(
                prompt="What is 2+2? Respond with just the number.",
                model_name="openai/gpt-4o",
                temperature=0.0,
                max_tokens=10
            )
            
            response = await client.generate_response(request)
            
            if response.success:
                print(f"✅ Test successful!")
                print(f"   Model: {response.model_name}")
                print(f"   Response: {response.response_text}")
                print(f"   Cost: ${response.cost_usd:.4f}")
                print(f"   Latency: {response.latency_seconds:.2f}s")
            else:
                print(f"❌ Test failed: {response.error_message}")
            
            print(f"\n📊 Usage Summary:")
            summary = client.get_usage_summary()
            for key, value in summary.items():
                print(f"   {key}: {value}")
    
    # Run test
    asyncio.run(test_client())