"""
Feature extraction pipeline for AMIL.
Combines code embeddings (GraphCodeBERT/CodeBERT) with handcrafted features.
"""

import math
import re
import ast
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging

from .config import AMILConfig

logger = logging.getLogger(__name__)


@dataclass
class UnitFeatures:
    """Features extracted from a single code unit (file/function)."""
    
    # Core identifiers
    unit_name: str
    file_path: str
    unit_type: str  # file, function, manifest
    ecosystem: str  # npm, pypi, cargo
    
    # Code content
    raw_content: str
    tokens: List[str] = field(default_factory=list)
    
    # Embeddings
    code_embedding: Optional[torch.Tensor] = None  # 768-dim from CodeBERT
    
    # API features (20-dim)
    api_counts: Dict[str, int] = field(default_factory=dict)
    api_features: Optional[torch.Tensor] = None
    
    # Entropy features (5-dim)
    shannon_entropy: float = 0.0
    string_entropy: float = 0.0
    obfuscation_score: float = 0.0
    base64_detected: bool = False
    high_entropy_ratio: float = 0.0
    entropy_features: Optional[torch.Tensor] = None
    
    # Phase features (3-dim one-hot)
    phase: str = "runtime"  # install, runtime, test
    phase_features: Optional[torch.Tensor] = None
    
    # Metadata features (10-dim)
    file_size_bytes: int = 0
    num_imports: int = 0
    num_functions: int = 0
    num_classes: int = 0
    cyclomatic_complexity: int = 0
    num_strings: int = 0
    num_comments: int = 0
    loc: int = 0  # Lines of code
    avg_line_length: float = 0.0
    import_entropy: float = 0.0  # Diversity of imports
    metadata_features: Optional[torch.Tensor] = None
    
    # Final fused features
    unit_embedding: Optional[torch.Tensor] = None  # 512-dim final embedding


class AMILFeatureExtractor(nn.Module):
    """
    Extract and fuse features for AMIL model.
    Combines pre-trained code embeddings with handcrafted features.
    """
    
    def __init__(self, config: AMILConfig, code_model_name: str = "microsoft/graphcodebert-base"):
        super().__init__()
        self.config = config
        
        # Load pre-trained code model (GraphCodeBERT preferred)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(code_model_name)
            self.code_model = AutoModel.from_pretrained(code_model_name)
            logger.info(f"Loaded code model: {code_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {code_model_name}, falling back to CodeBERT: {e}")
            code_model_name = "microsoft/codebert-base"
            self.tokenizer = AutoTokenizer.from_pretrained(code_model_name)
            self.code_model = AutoModel.from_pretrained(code_model_name)
        
        # Freeze code model initially (can unfreeze for fine-tuning)
        self.freeze_code_model()
        
        # Feature fusion network
        total_dim = config.total_feature_dim()
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, config.unit_embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.unit_embedding_dim * 2, config.unit_embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate / 2)
        )
        
        # Initialize API pattern matchers
        self._init_api_patterns()
        
    def freeze_code_model(self):
        """Freeze pre-trained code model parameters."""
        for param in self.code_model.parameters():
            param.requires_grad = False
            
    def unfreeze_code_model(self):
        """Unfreeze code model for fine-tuning."""
        for param in self.code_model.parameters():
            param.requires_grad = True
    
    def _init_api_patterns(self):
        """Initialize regex patterns for API call detection."""
        self.api_patterns = {
            # Network operations
            "net.outbound": [
                r"requests\.(get|post|put|delete|patch)",
                r"urllib\.request\.",
                r"http\.client\.",
                r"fetch\(",
                r"axios\.",
                r"XMLHttpRequest",
                r"\.get\s*\(",  # Generic GET
                r"\.post\s*\(",  # Generic POST
            ],
            "net.inbound": [
                r"socket\.listen",
                r"http\.server",
                r"express\(\)",
                r"app\.listen",
                r"server\.listen",
            ],
            
            # File system operations
            "fs.read": [
                r"open\s*\(",
                r"\.read\(\)",
                r"\.readFile",
                r"fs\.readFile",
                r"io\.open",
            ],
            "fs.write": [
                r"\.write\(",
                r"\.writeFile",
                r"fs\.writeFile",
                r"open\([^,]+,\s*[\"']w",
            ],
            "fs.delete": [
                r"os\.remove",
                r"os\.unlink",
                r"fs\.unlink",
                r"\.unlink\(",
                r"shutil\.rmtree",
            ],
            
            # Crypto operations
            "crypto.hash": [
                r"hashlib\.",
                r"crypto\.createHash",
                r"md5\(",
                r"sha1\(",
                r"sha256\(",
            ],
            "crypto.encrypt": [
                r"crypto\.createCipher",
                r"AES\.",
                r"RSA\.",
                r"\.encrypt\(",
            ],
            
            # Process operations  
            "subprocess.spawn": [
                r"subprocess\.",
                r"os\.system",
                r"os\.popen",
                r"child_process\.spawn",
                r"exec\(",
                r"execSync\(",
            ],
            "subprocess.shell": [
                r"\/bin\/sh",
                r"\/bin\/bash",
                r"cmd\.exe",
                r"powershell",
                r"shell=True",
            ],
            
            # Eval operations
            "eval.exec": [
                r"\beval\s*\(",
                r"\bexec\s*\(",
                r"Function\s*\(",
                r"new Function",
            ],
            "eval.compile": [
                r"compile\s*\(",
                r"\.compile\(",
                r"vm\.runInNewContext",
            ],
            
            # Environment access
            "env.read": [
                r"os\.environ",
                r"process\.env",
                r"getenv\(",
                r"\.env\.",
            ],
            "env.write": [
                r"os\.environ\[",
                r"process\.env\[",
                r"setenv\(",
            ],
            
            # Install-time hooks
            "install.hook": [
                r"postinstall",
                r"preinstall", 
                r"setup\.py",
                r"__init__\.py",
                r"install_requires",
            ],
            "install.script": [
                r"npm run",
                r"yarn run",
                r"pip install",
                r"python setup\.py",
            ],
            
            # Obfuscation indicators
            "obfuscation.base64": [
                r"base64\.",
                r"atob\(",
                r"btoa\(",
                r"Buffer\.from\([^,]+,\s*[\"']base64",
            ],
            "obfuscation.eval": [
                r"eval\s*\(\s*atob",
                r"eval\s*\(\s*Buffer",
                r"Function\s*\(\s*atob",
            ],
            
            # System access
            "registry.access": [
                r"winreg\.",
                r"Registry\.",
                r"regedit",
            ],
            "browser.access": [
                r"sqlite3",
                r"Login Data",
                r"Cookies",
                r"History",
                r"chrome",
                r"firefox",
            ],
            "system.info": [
                r"platform\.",
                r"os\.uname",
                r"sys\.version",
                r"navigator\.",
            ],
        }
    
    def extract_unit_features(self, raw_content: str, file_path: str, 
                             unit_name: str = "", unit_type: str = "file",
                             ecosystem: str = "unknown") -> UnitFeatures:
        """Extract all features for a single code unit."""
        
        features = UnitFeatures(
            unit_name=unit_name or Path(file_path).name,
            file_path=file_path,
            unit_type=unit_type,
            ecosystem=ecosystem,
            raw_content=raw_content
        )
        
        # Extract different feature types
        self._extract_api_features(features)
        self._extract_entropy_features(features)
        self._extract_phase_features(features)
        self._extract_metadata_features(features)
        
        return features
    
    def _extract_api_features(self, features: UnitFeatures):
        """Extract API call counts and patterns."""
        content = features.raw_content
        api_counts = {}
        
        for category, patterns in self.api_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                count += len(matches)
            api_counts[category] = count
        
        features.api_counts = api_counts
        
        # Convert to tensor (log-scaled as per AMIL.md)
        counts = [api_counts.get(cat, 0) for cat in self.config.api_categories]
        log_counts = [math.log(1 + count) for count in counts]
        features.api_features = torch.tensor(log_counts, dtype=torch.float32)
    
    def _extract_entropy_features(self, features: UnitFeatures):
        """Extract entropy and obfuscation features."""
        content = features.raw_content
        
        # Shannon entropy of entire content
        features.shannon_entropy = self._calculate_shannon_entropy(content)
        
        # String literal entropy
        strings = re.findall(r'["\']([^"\']{10,})["\']', content)
        if strings:
            avg_string_entropy = np.mean([self._calculate_shannon_entropy(s) for s in strings])
            features.string_entropy = avg_string_entropy
        
        # Obfuscation score (high entropy + suspicious patterns)
        obfuscation_indicators = [
            len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{20,}', content)),  # Very long identifiers
            len(re.findall(r'\\x[0-9a-fA-F]{2}', content)),  # Hex escapes
            len(re.findall(r'["\'][A-Za-z0-9+/]{50,}={0,2}["\']', content)),  # Base64-like
        ]
        features.obfuscation_score = sum(obfuscation_indicators) / max(len(content), 1)
        
        # Base64 detection
        features.base64_detected = bool(re.search(r'[A-Za-z0-9+/]{40,}={0,2}', content))
        
        # High entropy ratio
        lines = content.split('\n')
        high_entropy_lines = sum(1 for line in lines if self._calculate_shannon_entropy(line) > 5.0)
        features.high_entropy_ratio = high_entropy_lines / max(len(lines), 1)
        
        # Convert to tensor
        entropy_vals = [
            features.shannon_entropy / 8.0,  # Normalize to ~[0,1]
            features.string_entropy / 8.0,
            features.obfuscation_score,
            1.0 if features.base64_detected else 0.0,
            features.high_entropy_ratio
        ]
        features.entropy_features = torch.tensor(entropy_vals, dtype=torch.float32)
    
    def _extract_phase_features(self, features: UnitFeatures):
        """Determine execution phase (install/runtime/test)."""
        file_path = features.file_path.lower()
        content = features.raw_content.lower()
        
        # Install phase indicators
        if any(indicator in file_path for indicator in ['setup.py', 'postinstall', '__init__.py']):
            features.phase = "install"
        elif any(indicator in content for indicator in ['postinstall', 'preinstall', 'npm run']):
            features.phase = "install"
        # Test phase indicators
        elif any(indicator in file_path for indicator in ['test', 'spec', '__test__']):
            features.phase = "test"
        elif any(indicator in content for indicator in ['test(', 'it(', 'describe(']):
            features.phase = "test"
        # Default to runtime
        else:
            features.phase = "runtime"
        
        # One-hot encoding
        phase_map = {"install": 0, "runtime": 1, "test": 2}
        phase_vector = [0.0, 0.0, 0.0]
        phase_vector[phase_map[features.phase]] = 1.0
        features.phase_features = torch.tensor(phase_vector, dtype=torch.float32)
    
    def _extract_metadata_features(self, features: UnitFeatures):
        """Extract structural metadata features."""
        content = features.raw_content
        lines = content.split('\n')
        
        # Basic counts
        features.file_size_bytes = len(content)
        features.loc = len([line for line in lines if line.strip()])
        features.avg_line_length = np.mean([len(line) for line in lines]) if lines else 0
        
        # Language-specific parsing
        if features.ecosystem == "pypi" or features.file_path.endswith('.py'):
            self._extract_python_metadata(features, content)
        elif features.ecosystem == "npm" or features.file_path.endswith(('.js', '.ts')):
            self._extract_javascript_metadata(features, content)
        else:
            # Generic extraction
            self._extract_generic_metadata(features, content)
        
        # Import entropy (diversity of imports)
        import_names = re.findall(r'(?:import|require|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        if import_names:
            features.import_entropy = self._calculate_shannon_entropy(' '.join(import_names))
        
        # Convert to tensor (log-scaled for counts)
        metadata_vals = [
            math.log(1 + features.file_size_bytes) / 15,  # Normalize large files
            math.log(1 + features.num_imports),
            math.log(1 + features.num_functions),
            math.log(1 + features.num_classes),
            math.log(1 + features.cyclomatic_complexity),
            math.log(1 + features.num_strings),
            math.log(1 + features.num_comments),
            math.log(1 + features.loc),
            features.avg_line_length / 100,  # Normalize line length
            features.import_entropy / 8.0
        ]
        features.metadata_features = torch.tensor(metadata_vals, dtype=torch.float32)
    
    def _extract_python_metadata(self, features: UnitFeatures, content: str):
        """Extract Python-specific metadata."""
        try:
            tree = ast.parse(content)
            
            # Count constructs
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features.num_functions += 1
                elif isinstance(node, ast.ClassDef):
                    features.num_classes += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    features.num_imports += 1
                elif isinstance(node, ast.Str):
                    features.num_strings += 1
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    features.cyclomatic_complexity += 1
                    
        except SyntaxError:
            # Fallback to regex if AST parsing fails
            self._extract_generic_metadata(features, content)
    
    def _extract_javascript_metadata(self, features: UnitFeatures, content: str):
        """Extract JavaScript-specific metadata."""
        # Function declarations
        features.num_functions = len(re.findall(r'function\s+\w+|=>\s*{|\w+\s*:\s*function', content))
        
        # Class declarations
        features.num_classes = len(re.findall(r'class\s+\w+', content))
        
        # Imports/requires
        imports = re.findall(r'(?:import|require)\s*\(|from\s+["\']', content)
        features.num_imports = len(imports)
        
        # String literals
        features.num_strings = len(re.findall(r'["\'][^"\']*["\']', content))
        
        # Comments
        features.num_comments = len(re.findall(r'//.*|/\*.*?\*/', content, re.DOTALL))
        
        # Control flow complexity
        complexity_patterns = [r'\bif\s*\(', r'\bfor\s*\(', r'\bwhile\s*\(', r'\btry\s*{', r'\bcatch\s*\(']
        features.cyclomatic_complexity = sum(len(re.findall(pattern, content)) for pattern in complexity_patterns)
    
    def _extract_generic_metadata(self, features: UnitFeatures, content: str):
        """Generic metadata extraction using regex."""
        # Generic patterns
        features.num_functions = len(re.findall(r'def\s+\w+|function\s+\w+|\w+\s*\([^)]*\)\s*{', content))
        features.num_classes = len(re.findall(r'class\s+\w+', content))
        features.num_imports = len(re.findall(r'import\s+|require\s*\(|#include', content))
        features.num_strings = len(re.findall(r'["\'][^"\']*["\']', content))
        features.num_comments = len(re.findall(r'#.*|//.*|/\*.*?\*/', content, re.DOTALL))
        features.cyclomatic_complexity = len(re.findall(r'\b(?:if|for|while|try|catch)\b', content))
    
    def _calculate_shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        return entropy
    
    def get_code_embeddings(self, features_list: List[UnitFeatures], 
                           max_length: int = 512) -> torch.Tensor:
        """Extract code embeddings using pre-trained model."""
        
        # Prepare batch of code snippets
        code_snippets = []
        for features in features_list:
            # Truncate very long code for efficiency
            content = features.raw_content[:max_length * 10]  # Rough character limit
            code_snippets.append(content)
        
        # Tokenize batch
        tokenized = self.tokenizer(
            code_snippets,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move tokenized inputs to same device as model
        device = next(self.code_model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        # Extract embeddings
        with torch.no_grad():
            outputs = self.code_model(**tokenized)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)
        
        return embeddings
    
    def fuse_features(self, features: UnitFeatures) -> torch.Tensor:
        """Fuse all feature types into final unit embedding."""
        
        # Ensure code embedding exists
        if features.code_embedding is None:
            # Single unit embedding
            embeddings = self.get_code_embeddings([features])
            features.code_embedding = embeddings[0]
        
        # Concatenate all features
        feature_tensor = torch.cat([
            features.code_embedding,      # 768-dim
            features.api_features,        # 20-dim
            features.entropy_features,    # 5-dim  
            features.phase_features,      # 3-dim
            features.metadata_features    # 10-dim
        ], dim=0)
        
        # Project through fusion network
        features.unit_embedding = self.feature_fusion(feature_tensor)
        
        return features.unit_embedding
    
    def forward(self, features_list: List[UnitFeatures]) -> torch.Tensor:
        """Forward pass: extract and fuse features for batch of units."""
        
        # Extract code embeddings for entire batch
        code_embeddings = self.get_code_embeddings(features_list)
        
        # Update features with embeddings
        for i, features in enumerate(features_list):
            features.code_embedding = code_embeddings[i]
        
        # Fuse features for each unit
        unit_embeddings = []
        for features in features_list:
            embedding = self.fuse_features(features)
            unit_embeddings.append(embedding)
        
        return torch.stack(unit_embeddings, dim=0)


def extract_features_from_code(raw_content: str, file_path: str, 
                              config: AMILConfig, extractor: AMILFeatureExtractor) -> UnitFeatures:
    """Convenience function to extract features from raw code."""
    
    # Determine ecosystem from file path
    ecosystem = "unknown"
    if file_path.endswith('.py'):
        ecosystem = "pypi"
    elif file_path.endswith(('.js', '.ts', '.json')):
        ecosystem = "npm"
    elif file_path.endswith('.rs'):
        ecosystem = "cargo"
    
    return extractor.extract_unit_features(
        raw_content=raw_content,
        file_path=file_path,
        ecosystem=ecosystem
    )