"""
Benign package collection from npm and PyPI registries.
Collects packages with high downloads (popular) and long tail for diversity.
"""

import asyncio
import json
import os
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import requests
import aiohttp
import time


@dataclass 
class BenignSample:
    """Represents a benign package sample."""
    name: str
    ecosystem: str  # npm, pypi
    version: str
    download_count: Optional[int]
    category: str  # popular, longtail
    download_url: str
    extracted_path: Optional[Path] = None
    metadata: Optional[Dict] = None


class NpmCollector:
    """Collects benign packages from npm registry."""
    
    def __init__(self):
        self.registry_url = "https://registry.npmjs.org"
        self.downloads_api = "https://api.npmjs.org/downloads"
        
    def get_popular_packages(self, limit: int = 1000) -> List[str]:
        """Get popular packages by download count."""
        # Use npm's download statistics
        url = f"{self.downloads_api}/point/last-month"
        
        # Get top packages from various categories
        popular_names = []
        
        # Method 1: Use known popular packages as seeds
        seed_packages = [
            "react", "vue", "angular", "lodash", "express", "request", "chalk",
            "commander", "inquirer", "fs-extra", "axios", "moment", "underscore",
            "async", "debug", "yargs", "glob", "rimraf", "mkdirp", "semver",
            "colors", "through2", "readable-stream", "minimist", "bluebird",
            "uuid", "babel-core", "webpack", "eslint", "mocha", "jest"
        ]
        
        # Get their dependencies and dependents for more popular packages
        for pkg in seed_packages[:20]:  # Limit to avoid rate limiting
            try:
                deps = self._get_package_dependencies(pkg)
                popular_names.extend(deps)
                if len(popular_names) >= limit:
                    break
                time.sleep(0.1)  # Rate limiting
            except:
                continue
                
        return list(set(popular_names))[:limit]
    
    def get_longtail_packages(self, limit: int = 5000, exclude: Set[str] = None) -> List[str]:
        """Get long tail packages by sampling recent publications."""
        if exclude is None:
            exclude = set()
            
        # Search for packages with various keywords to get diversity
        keywords = [
            "utility", "helper", "tool", "cli", "parser", "formatter", "validator",
            "converter", "generator", "wrapper", "client", "sdk", "api", "plugin",
            "middleware", "component", "library", "framework", "template", "config"
        ]
        
        longtail_names = []
        for keyword in keywords:
            try:
                # Search npm registry
                search_url = f"https://registry.npmjs.org/-/v1/search"
                params = {
                    "text": keyword,
                    "size": min(50, limit // len(keywords)),
                    "from": 0
                }
                
                response = requests.get(search_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for obj in data.get("objects", []):
                        pkg = obj.get("package", {})
                        name = pkg.get("name")
                        if name and name not in exclude:
                            longtail_names.append(name)
                
                time.sleep(0.1)  # Rate limiting
                if len(longtail_names) >= limit:
                    break
                    
            except Exception as e:
                print(f"Error searching for keyword '{keyword}': {e}")
                continue
        
        return list(set(longtail_names))[:limit]
    
    def _get_package_dependencies(self, package_name: str) -> List[str]:
        """Get dependencies of a package."""
        try:
            url = f"{self.registry_url}/{package_name}/latest"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                deps = list(data.get("dependencies", {}).keys())
                dev_deps = list(data.get("devDependencies", {}).keys())
                return deps + dev_deps
        except:
            pass
        return []
    
    def get_package_info(self, package_name: str) -> Optional[BenignSample]:
        """Get package metadata and create BenignSample."""
        try:
            # Get package metadata
            url = f"{self.registry_url}/{package_name}"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            # Get latest version info
            latest_version = data.get("dist-tags", {}).get("latest")
            if not latest_version:
                return None
                
            version_data = data.get("versions", {}).get(latest_version, {})
            tarball_url = version_data.get("dist", {}).get("tarball")
            
            if not tarball_url:
                return None
            
            # Try to get download count
            download_count = None
            try:
                downloads_url = f"{self.downloads_api}/point/last-month/{package_name}"
                dl_response = requests.get(downloads_url, timeout=5)
                if dl_response.status_code == 200:
                    download_count = dl_response.json().get("downloads")
            except:
                pass
            
            # Categorize as popular vs longtail based on download count
            category = "popular" if download_count and download_count > 10000 else "longtail"
            
            return BenignSample(
                name=package_name,
                ecosystem="npm",
                version=latest_version,
                download_count=download_count,
                category=category,
                download_url=tarball_url,
                metadata=version_data
            )
            
        except Exception as e:
            print(f"Error getting package info for {package_name}: {e}")
            return None
    
    def download_package(self, sample: BenignSample, output_dir: Path) -> bool:
        """Download and extract npm package."""
        try:
            # Create package directory
            pkg_dir = output_dir / sample.ecosystem / sample.category / sample.name / sample.version
            pkg_dir.mkdir(parents=True, exist_ok=True)
            
            # Download tarball
            response = requests.get(sample.download_url, stream=True, timeout=30)
            if response.status_code != 200:
                return False
            
            # Save and extract tarball
            tarball_path = pkg_dir / f"{sample.name}-{sample.version}.tgz"
            with open(tarball_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract tarball
            with tarfile.open(tarball_path, 'r:gz') as tar:
                tar.extractall(pkg_dir)
                
            # Clean up tarball
            tarball_path.unlink()
            
            # Update sample with extracted path
            sample.extracted_path = pkg_dir
            return True
            
        except Exception as e:
            print(f"Error downloading {sample.name}: {e}")
            return False


class PyPICollector:
    """Collects benign packages from PyPI."""
    
    def __init__(self):
        self.api_url = "https://pypi.org/pypi"
        self.simple_url = "https://pypi.org/simple"
        
    def get_popular_packages(self, limit: int = 1000) -> List[str]:
        """Get popular PyPI packages."""
        # Use known popular packages
        popular_seeds = [
            "requests", "urllib3", "setuptools", "certifi", "charset-normalizer",
            "idna", "pip", "wheel", "six", "python-dateutil", "numpy", "pyyaml",
            "click", "jinja2", "markupsafe", "packaging", "pyparsing", "pytz",
            "colorama", "cffi", "pycparser", "cryptography", "attrs", "jsonschema",
            "importlib-metadata", "zipp", "typing-extensions", "platformdirs",
            "tomli", "filelock", "distlib", "virtualenv", "tqdm", "pillow",
            "pandas", "scipy", "matplotlib", "django", "flask", "fastapi"
        ]
        
        return popular_seeds[:limit]
    
    def get_longtail_packages(self, limit: int = 5000, exclude: Set[str] = None) -> List[str]:
        """Get long tail packages from PyPI simple index."""
        if exclude is None:
            exclude = set()
            
        try:
            # Get packages from PyPI simple index
            response = requests.get(f"{self.simple_url}/", timeout=60)
            if response.status_code != 200:
                return []
            
            # Parse HTML to extract package names
            import re
            pattern = r'<a href="[^"]*">([^<]+)</a>'
            matches = re.findall(pattern, response.text)
            
            # Filter and sample
            longtail = [name for name in matches if name not in exclude]
            
            # Sample evenly to get diversity
            step = max(1, len(longtail) // limit)
            return longtail[::step][:limit]
            
        except Exception as e:
            print(f"Error getting PyPI packages: {e}")
            return []
    
    def get_package_info(self, package_name: str) -> Optional[BenignSample]:
        """Get PyPI package metadata."""
        try:
            url = f"{self.api_url}/{package_name}/json"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
                
            data = response.json()
            info = data.get("info", {})
            
            # Get latest version
            version = info.get("version")
            if not version:
                return None
            
            # Find source distribution or wheel
            releases = data.get("releases", {}).get(version, [])
            download_url = None
            
            # Prefer source distribution (.tar.gz)
            for release in releases:
                if release.get("filename", "").endswith(".tar.gz"):
                    download_url = release.get("url")
                    break
            
            # Fall back to wheel
            if not download_url:
                for release in releases:
                    if release.get("filename", "").endswith(".whl"):
                        download_url = release.get("url")
                        break
            
            if not download_url:
                return None
            
            # Estimate category based on GitHub stars or description
            category = "longtail"  # Default to longtail
            
            return BenignSample(
                name=package_name,
                ecosystem="pypi", 
                version=version,
                download_count=None,  # PyPI doesn't provide download stats easily
                category=category,
                download_url=download_url,
                metadata=info
            )
            
        except Exception as e:
            print(f"Error getting PyPI package info for {package_name}: {e}")
            return None
    
    def download_package(self, sample: BenignSample, output_dir: Path) -> bool:
        """Download and extract PyPI package."""
        try:
            # Create package directory
            pkg_dir = output_dir / sample.ecosystem / sample.category / sample.name / sample.version
            pkg_dir.mkdir(parents=True, exist_ok=True)
            
            # Download package
            response = requests.get(sample.download_url, stream=True, timeout=30)
            if response.status_code != 200:
                return False
            
            filename = sample.download_url.split("/")[-1]
            file_path = pkg_dir / filename
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract based on file type
            if filename.endswith('.tar.gz'):
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(pkg_dir)
            elif filename.endswith('.whl'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(pkg_dir)
            else:
                print(f"Unknown file type for {filename}")
                return False
            
            # Clean up downloaded file
            file_path.unlink()
            
            sample.extracted_path = pkg_dir
            return True
            
        except Exception as e:
            print(f"Error downloading PyPI package {sample.name}: {e}")
            return False


class BenignCollector:
    """Main collector that coordinates npm and PyPI collection."""
    
    def __init__(self):
        self.npm = NpmCollector()
        self.pypi = PyPICollector()
    
    def collect_balanced_dataset(self, 
                                total_samples: int = 50000,
                                npm_ratio: float = 0.7,
                                popular_ratio: float = 0.5,
                                output_dir: Optional[Path] = None) -> List[BenignSample]:
        """Collect balanced dataset of benign packages."""
        
        if output_dir is None:
            output_dir = Path("benign_packages")
        
        npm_samples = int(total_samples * npm_ratio)
        pypi_samples = total_samples - npm_samples
        
        collected = []
        
        # Collect npm packages
        print(f"Collecting {npm_samples} npm packages...")
        npm_popular_count = int(npm_samples * popular_ratio)
        npm_longtail_count = npm_samples - npm_popular_count
        
        # Get popular npm packages
        npm_popular_names = self.npm.get_popular_packages(npm_popular_count * 2)  # Get extra for filtering
        npm_collected = 0
        
        for name in npm_popular_names[:npm_popular_count * 2]:  # Try more than needed
            if npm_collected >= npm_popular_count:
                break
                
            sample = self.npm.get_package_info(name)
            if sample:
                sample.category = "popular"
                if self.npm.download_package(sample, output_dir):
                    collected.append(sample)
                    npm_collected += 1
                    if npm_collected % 50 == 0:
                        print(f"  Collected {npm_collected} popular npm packages...")
            
            time.sleep(0.1)  # Rate limiting
        
        # Get longtail npm packages
        npm_popular_set = set(s.name for s in collected if s.ecosystem == "npm")
        npm_longtail_names = self.npm.get_longtail_packages(npm_longtail_count * 3, exclude=npm_popular_set)
        npm_longtail_collected = 0
        
        for name in npm_longtail_names:
            if npm_longtail_collected >= npm_longtail_count:
                break
                
            sample = self.npm.get_package_info(name)
            if sample:
                sample.category = "longtail"
                if self.npm.download_package(sample, output_dir):
                    collected.append(sample)
                    npm_longtail_collected += 1
                    if npm_longtail_collected % 50 == 0:
                        print(f"  Collected {npm_longtail_collected} longtail npm packages...")
            
            time.sleep(0.1)  # Rate limiting
        
        # Collect PyPI packages
        print(f"Collecting {pypi_samples} PyPI packages...")
        pypi_popular_count = int(pypi_samples * popular_ratio)
        pypi_longtail_count = pypi_samples - pypi_popular_count
        
        # Get popular PyPI packages
        pypi_popular_names = self.pypi.get_popular_packages(pypi_popular_count * 2)
        pypi_collected = 0
        
        for name in pypi_popular_names:
            if pypi_collected >= pypi_popular_count:
                break
                
            sample = self.pypi.get_package_info(name)
            if sample:
                sample.category = "popular"
                if self.pypi.download_package(sample, output_dir):
                    collected.append(sample)
                    pypi_collected += 1
                    if pypi_collected % 50 == 0:
                        print(f"  Collected {pypi_collected} popular PyPI packages...")
            
            time.sleep(0.1)  # Rate limiting
        
        # Get longtail PyPI packages
        pypi_popular_set = set(s.name for s in collected if s.ecosystem == "pypi")
        pypi_longtail_names = self.pypi.get_longtail_packages(pypi_longtail_count * 2, exclude=pypi_popular_set)
        pypi_longtail_collected = 0
        
        for name in pypi_longtail_names:
            if pypi_longtail_collected >= pypi_longtail_count:
                break
                
            sample = self.pypi.get_package_info(name)
            if sample:
                sample.category = "longtail"
                if self.pypi.download_package(sample, output_dir):
                    collected.append(sample)
                    pypi_longtail_collected += 1
                    if pypi_longtail_collected % 50 == 0:
                        print(f"  Collected {pypi_longtail_collected} longtail PyPI packages...")
            
            time.sleep(0.1)  # Rate limiting
        
        print(f"Collection complete! Total: {len(collected)} packages")
        print(f"  npm: {len([s for s in collected if s.ecosystem == 'npm'])}")
        print(f"  pypi: {len([s for s in collected if s.ecosystem == 'pypi'])}")
        print(f"  popular: {len([s for s in collected if s.category == 'popular'])}")
        print(f"  longtail: {len([s for s in collected if s.category == 'longtail'])}")
        
        return collected


if __name__ == "__main__":
    # Test basic functionality
    collector = BenignCollector()
    
    # Test npm collection
    print("Testing npm popular packages...")
    npm_popular = collector.npm.get_popular_packages(5)
    print(f"Found {len(npm_popular)} popular npm packages: {npm_popular}")
    
    # Test PyPI collection
    print("\nTesting PyPI popular packages...")
    pypi_popular = collector.pypi.get_popular_packages(5)
    print(f"Found {len(pypi_popular)} popular PyPI packages: {pypi_popular}")
    
    # Test package info
    print("\nTesting package info...")
    if npm_popular:
        sample = collector.npm.get_package_info(npm_popular[0])
        if sample:
            print(f"npm sample: {sample.name} v{sample.version} ({sample.download_count} downloads)")
    
    if pypi_popular:
        sample = collector.pypi.get_package_info(pypi_popular[0])
        if sample:
            print(f"PyPI sample: {sample.name} v{sample.version}")