"""
Unit processing for NeoBERT: splitting packages into analyzable units.
"""

import ast
import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
from transformers import AutoTokenizer

from .config import NeoBERTConfig

logger = logging.getLogger(__name__)


@dataclass
class PackageUnit:
    """Represents a unit of code for NeoBERT analysis."""
    
    # Basic identification
    unit_id: str  # Unique identifier
    unit_name: str  # Human-readable name
    unit_type: str  # "file", "function", "chunk"
    source_file: str
    
    # Content
    raw_content: str
    tokens: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    
    # Metadata
    line_start: int = 0
    line_end: int = 0
    char_start: int = 0
    char_end: int = 0
    
    # Engineered features
    risky_api_counts: Dict[str, int] = field(default_factory=dict)
    shannon_entropy: float = 0.0
    phase_tag: str = "runtime"  # "install", "runtime", "test"
    file_size: int = 0
    import_count: int = 0
    
    # Tokenization info
    is_truncated: bool = False
    chunk_index: int = 0  # For multi-chunk units
    total_chunks: int = 1
    
    def __post_init__(self):
        """Calculate derived features."""
        self.file_size = len(self.raw_content)
        self.shannon_entropy = self._calculate_entropy()
        self.import_count = self._count_imports()
    
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of the content."""
        if not self.raw_content:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in self.raw_content:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        text_len = len(self.raw_content)
        entropy = 0.0
        for count in char_counts.values():
            prob = count / text_len
            entropy -= prob * math.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(char_counts)) if char_counts else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _count_imports(self) -> int:
        """Count import statements."""
        import_patterns = [
            r'^import\s+',
            r'^from\s+.*\s+import\s+',
            r'require\s*\(',
            r'#include\s*[<"]',
            r'using\s+.*::',
        ]
        
        count = 0
        lines = self.raw_content.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in import_patterns:
                if re.match(pattern, line):
                    count += 1
                    break
        
        return count
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of extracted features."""
        return {
            "unit_type": self.unit_type,
            "size_bytes": self.file_size,
            "entropy": self.shannon_entropy,
            "phase": self.phase_tag,
            "imports": self.import_count,
            "risky_apis": sum(self.risky_api_counts.values()),
            "tokens": len(self.token_ids),
            "is_truncated": self.is_truncated,
            "chunks": self.total_chunks
        }


class UnitProcessor:
    """Processes packages into units for NeoBERT analysis."""
    
    def __init__(self, config: NeoBERTConfig):
        self.config = config
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {config.model_name}: {e}")
            logger.info("Falling back to CodeBERT tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Compile API patterns for efficiency
        self.api_patterns = self._compile_api_patterns()
        
        # Phase detection patterns
        self.install_patterns = [re.compile(p, re.IGNORECASE) for p in config.install_phase_patterns]
        self.test_patterns = [re.compile(p, re.IGNORECASE) for p in config.test_phase_patterns]
    
    def process_package(self, 
                       package_name: str,
                       file_contents: Dict[str, str],
                       ecosystem: str = "unknown") -> List[PackageUnit]:
        """Process a package into units."""
        
        units = []
        
        # Process each file
        for file_path, content in file_contents.items():
            if not content or not content.strip():
                continue
            
            file_units = self._process_file(
                file_path, 
                content, 
                package_name,
                ecosystem
            )
            units.extend(file_units)
        
        # Limit total units
        if len(units) > self.config.max_units_per_package:
            units = self._select_most_important_units(units)
        
        logger.debug(f"Processed package {package_name}: {len(units)} units")
        return units
    
    def _process_file(self, 
                     file_path: str,
                     content: str,
                     package_name: str,
                     ecosystem: str) -> List[PackageUnit]:
        """Process a single file into units."""
        
        if self.config.unit_type == "file":
            return self._process_file_as_units(file_path, content, package_name, ecosystem)
        elif self.config.unit_type == "function":
            return self._process_file_as_functions(file_path, content, package_name, ecosystem)
        else:  # mixed
            # Try functions first, fall back to file if extraction fails
            try:
                func_units = self._process_file_as_functions(file_path, content, package_name, ecosystem)
                if func_units:
                    return func_units
            except Exception as e:
                logger.debug(f"Function extraction failed for {file_path}: {e}")
            
            return self._process_file_as_units(file_path, content, package_name, ecosystem)
    
    def _process_file_as_units(self, 
                              file_path: str,
                              content: str,
                              package_name: str,
                              ecosystem: str) -> List[PackageUnit]:
        """Process entire file as single unit, with chunking if needed."""
        
        # Tokenize content
        tokens = self.tokenizer.tokenize(content)
        
        units = []
        
        if len(tokens) <= self.config.max_tokens_per_unit:
            # Single unit
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            unit = PackageUnit(
                unit_id=self._generate_unit_id(package_name, file_path, 0),
                unit_name=Path(file_path).name,
                unit_type="file",
                source_file=file_path,
                raw_content=content,
                tokens=tokens,
                token_ids=token_ids,
                line_end=len(content.split('\n'))
            )
            
            self._extract_features(unit, ecosystem)
            units.append(unit)
            
        else:
            # Multiple chunks needed
            units = self._create_chunks(
                file_path, content, tokens, package_name, ecosystem, "file"
            )
        
        return units
    
    def _process_file_as_functions(self,
                                 file_path: str,
                                 content: str,
                                 package_name: str,
                                 ecosystem: str) -> List[PackageUnit]:
        """Extract functions from file as individual units."""
        
        functions = []
        
        try:
            if ecosystem == "pypi" or file_path.endswith('.py'):
                functions = self._extract_python_functions(content)
            elif ecosystem == "npm" or file_path.endswith(('.js', '.ts')):
                functions = self._extract_javascript_functions(content)
        except Exception as e:
            logger.debug(f"Function extraction failed for {file_path}: {e}")
            return []
        
        units = []
        
        for func_name, func_content, line_start, line_end in functions:
            # Skip very small functions
            if len(func_content.strip()) < self.config.min_unit_tokens:
                continue
            
            # Tokenize function
            tokens = self.tokenizer.tokenize(func_content)
            
            if len(tokens) <= self.config.max_tokens_per_unit:
                # Single function unit
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                
                unit = PackageUnit(
                    unit_id=self._generate_unit_id(package_name, file_path, len(units)),
                    unit_name=f"{Path(file_path).stem}.{func_name}",
                    unit_type="function",
                    source_file=file_path,
                    raw_content=func_content,
                    tokens=tokens,
                    token_ids=token_ids,
                    line_start=line_start,
                    line_end=line_end
                )
                
                self._extract_features(unit, ecosystem)
                units.append(unit)
                
            else:
                # Large function needs chunking
                func_chunks = self._create_chunks(
                    file_path, func_content, tokens, package_name, ecosystem, "function"
                )
                units.extend(func_chunks)
        
        return units
    
    def _create_chunks(self,
                      file_path: str,
                      content: str,
                      tokens: List[str],
                      package_name: str,
                      ecosystem: str,
                      unit_type: str) -> List[PackageUnit]:
        """Create chunks from large content."""
        
        chunks = []
        max_tokens = self.config.max_tokens_per_unit
        overlap = self.config.chunk_overlap
        
        if self.config.unit_chunking_strategy == "truncate":
            # Simple truncation
            truncated_tokens = tokens[:max_tokens]
            token_ids = self.tokenizer.convert_tokens_to_ids(truncated_tokens)
            
            unit = PackageUnit(
                unit_id=self._generate_unit_id(package_name, file_path, 0),
                unit_name=f"{Path(file_path).name}_chunk_0",
                unit_type=unit_type,
                source_file=file_path,
                raw_content=self.tokenizer.convert_tokens_to_string(truncated_tokens),
                tokens=truncated_tokens,
                token_ids=token_ids,
                is_truncated=True
            )
            
            self._extract_features(unit, ecosystem)
            chunks.append(unit)
            
        else:
            # Sliding window chunking
            start = 0
            chunk_index = 0
            total_chunks = math.ceil((len(tokens) - overlap) / (max_tokens - overlap))
            
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                
                if len(chunk_tokens) < self.config.min_unit_tokens:
                    break
                
                token_ids = self.tokenizer.convert_tokens_to_ids(chunk_tokens)
                chunk_content = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                
                unit = PackageUnit(
                    unit_id=self._generate_unit_id(package_name, file_path, chunk_index),
                    unit_name=f"{Path(file_path).name}_chunk_{chunk_index}",
                    unit_type=unit_type,
                    source_file=file_path,
                    raw_content=chunk_content,
                    tokens=chunk_tokens,
                    token_ids=token_ids,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    is_truncated=(end < len(tokens))
                )
                
                self._extract_features(unit, ecosystem)
                chunks.append(unit)
                
                start += max_tokens - overlap
                chunk_index += 1
        
        return chunks
    
    def _extract_python_functions(self, content: str) -> List[Tuple[str, str, int, int]]:
        """Extract Python functions from content."""
        
        functions = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    
                    # Get function source
                    start_line = node.lineno - 1  # 0-indexed
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    func_lines = lines[start_line:end_line]
                    func_content = '\n'.join(func_lines)
                    
                    functions.append((func_name, func_content, start_line + 1, end_line))
                    
        except SyntaxError as e:
            logger.debug(f"Syntax error parsing Python code: {e}")
        
        return functions
    
    def _extract_javascript_functions(self, content: str) -> List[Tuple[str, str, int, int]]:
        """Extract JavaScript functions from content."""
        
        functions = []
        lines = content.split('\n')
        
        # Simple regex-based extraction for JavaScript functions
        patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*=\s*function\s*\(',
            r'(\w+)\s*=>\s*\{',
            r'const\s+(\w+)\s*=\s*\(',
            r'let\s+(\w+)\s*=\s*\(',
            r'var\s+(\w+)\s*=\s*\('
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    
                    # Extract function body (simplified)
                    start_line = i
                    end_line = min(i + 20, len(lines))  # Approximate
                    
                    func_lines = lines[start_line:end_line]
                    func_content = '\n'.join(func_lines)
                    
                    functions.append((func_name, func_content, start_line + 1, end_line))
                    break
        
        return functions
    
    def _extract_features(self, unit: PackageUnit, ecosystem: str):
        """Extract engineered features from unit."""
        
        # API analysis
        unit.risky_api_counts = self._count_risky_apis(unit.raw_content)
        
        # Phase detection
        unit.phase_tag = self._detect_phase(unit.source_file, unit.raw_content)
        
        # Additional metadata already computed in PackageUnit.__post_init__
    
    def _count_risky_apis(self, content: str) -> Dict[str, int]:
        """Count occurrences of risky APIs."""
        
        api_counts = {}
        
        for api, pattern in self.api_patterns.items():
            matches = pattern.findall(content)
            api_counts[api] = len(matches)
        
        return api_counts
    
    def _detect_phase(self, file_path: str, content: str) -> str:
        """Detect execution phase (install/runtime/test)."""
        
        # Check file path patterns
        for pattern in self.install_patterns:
            if pattern.search(file_path):
                return "install"
        
        for pattern in self.test_patterns:
            if pattern.search(file_path):
                return "test"
        
        # Check content patterns
        install_keywords = ["postinstall", "preinstall", "setup", "install"]
        test_keywords = ["test", "spec", "describe", "it(", "assert"]
        
        content_lower = content.lower()
        
        install_score = sum(content_lower.count(keyword) for keyword in install_keywords)
        test_score = sum(content_lower.count(keyword) for keyword in test_keywords)
        
        if install_score > test_score and install_score > 0:
            return "install"
        elif test_score > 0:
            return "test"
        else:
            return "runtime"
    
    def _compile_api_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for risky APIs."""
        
        patterns = {}
        
        for api in self.config.risky_apis:
            # Create pattern that matches API calls/imports
            pattern_str = rf'\b{re.escape(api)}\b'
            patterns[api] = re.compile(pattern_str, re.IGNORECASE)
        
        return patterns
    
    def _select_most_important_units(self, units: List[PackageUnit]) -> List[PackageUnit]:
        """Select most important units when over limit."""
        
        # Score units by importance
        scored_units = []
        
        for unit in units:
            score = 0
            
            # Risky API score
            score += sum(unit.risky_api_counts.values()) * 10
            
            # Size score (medium-sized units are preferred)
            size_score = min(unit.file_size / 1000, 10)
            if 100 <= unit.file_size <= 5000:  # Sweet spot
                size_score *= 2
            score += size_score
            
            # Phase score (install phase is more suspicious)
            if unit.phase_tag == "install":
                score += 20
            elif unit.phase_tag == "runtime":
                score += 10
            
            # Entropy score (higher entropy = more suspicious)
            score += unit.shannon_entropy * 5
            
            # Import score (many imports = more complex/important)
            score += unit.import_count
            
            scored_units.append((score, unit))
        
        # Sort by score and take top units
        scored_units.sort(key=lambda x: x[0], reverse=True)
        selected_units = [unit for score, unit in scored_units[:self.config.max_units_per_package]]
        
        logger.debug(f"Selected {len(selected_units)} most important units")
        return selected_units
    
    def _generate_unit_id(self, package_name: str, file_path: str, index: int) -> str:
        """Generate unique unit ID."""
        content = f"{package_name}:{file_path}:{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_processing_stats(self, units: List[PackageUnit]) -> Dict[str, Any]:
        """Get statistics about processed units."""
        
        if not units:
            return {"total_units": 0}
        
        stats = {
            "total_units": len(units),
            "unit_types": {},
            "phase_distribution": {},
            "size_stats": {
                "min": min(u.file_size for u in units),
                "max": max(u.file_size for u in units),
                "mean": np.mean([u.file_size for u in units]),
                "std": np.std([u.file_size for u in units])
            },
            "entropy_stats": {
                "min": min(u.shannon_entropy for u in units),
                "max": max(u.shannon_entropy for u in units),
                "mean": np.mean([u.shannon_entropy for u in units])
            },
            "total_risky_apis": sum(sum(u.risky_api_counts.values()) for u in units),
            "truncated_units": sum(1 for u in units if u.is_truncated),
            "chunked_units": sum(1 for u in units if u.total_chunks > 1)
        }
        
        # Count by type and phase
        for unit in units:
            stats["unit_types"][unit.unit_type] = stats["unit_types"].get(unit.unit_type, 0) + 1
            stats["phase_distribution"][unit.phase_tag] = stats["phase_distribution"].get(unit.phase_tag, 0) + 1
        
        return stats