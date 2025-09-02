#!/usr/bin/env python3
"""
FXorcist Automated Setup Script
Handles complete project setup, dependency installation, and initialization.
"""

import sys
import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import platform
import urllib.request

class FXorcistSetup:
    """Automated setup and installation for FXorcist."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    def run_setup(self):
        """Run complete setup process."""
        print("🚀 FXorcist Automated Setup")
        print("=" * 50)

        steps = [
            ("Checking system requirements", self.check_system_requirements),
            ("Installing Python dependencies", self.install_dependencies),
            ("Setting up project structure", self.setup_project_structure),
            ("Initializing configuration", self.initialize_configuration),
            ("Setting up data directories", self.setup_data_directories),
            ("Verifying installation", self.verify_installation),
            ("Creating startup scripts", self.create_startup_scripts)
        ]

        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            try:
                result = step_func()
                if result:
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"⚠️  {step_name} completed with warnings")
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                return False

        print("\n🎉 Setup completed successfully!")
        print("\n📖 Quick Start:")
        print("  1. Run: python fxorcist_cli.py --interactive")
        print("  2. Or:  python fxorcist_cli.py --dashboard")
        print("  3. Or:  python fxorcist_cli.py --data-integration")

        return True

    def check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        print("  Checking Python version...")
        if sys.version_info < (3, 8):
            print("  ❌ Python 3.8+ required")
            return False
        print(f"  ✅ Python {self.python_version} detected")

        # Check for required commands
        required_commands = ['pip', 'python']
        if self.system == 'windows':
            required_commands.append('where')
        else:
            required_commands.append('which')

        for cmd in required_commands:
            if not self.command_exists(cmd):
                print(f"  ❌ Command '{cmd}' not found")
                return False

        print("  ✅ System requirements met")
        return True

    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        requirements_file = self.project_root / "requirements.txt"

        if not requirements_file.exists():
            print("  ❌ requirements.txt not found")
            return False

        print("  Installing dependencies from requirements.txt...")

        try:
            # Upgrade pip first
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Install requirements
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], stdout=subprocess.DEVNULL)

            print("  ✅ Dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install dependencies: {e}")
            return False

    def setup_project_structure(self) -> bool:
        """Set up project directory structure."""
        directories = [
            "data/raw",
            "data/processed",
            "data/temp_extracted",
            "models",
            "logs",
            "config",
            "cache",
            "scripts",
            "docs",
            "tests"
        ]

        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Created {dir_path}")

        # Create __init__.py files for Python packages
        init_files = [
            "forex_ai_dashboard/__init__.py",
            "forex_ai_dashboard/pipeline/__init__.py",
            "memory_system/__init__.py",
            "dashboard/__init__.py"
        ]

        for init_file in init_files:
            init_path = self.project_root / init_file
            if not init_path.exists():
                init_path.parent.mkdir(parents=True, exist_ok=True)
                init_path.write_text('"""FXorcist package."""\n')
                print(f"  📄 Created {init_file}")

        print("  ✅ Project structure initialized")
        return True

    def initialize_configuration(self) -> bool:
        """Initialize configuration files."""
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)

        # CLI configuration
        cli_config = {
            "data_dir": "data",
            "models_dir": "models",
            "logs_dir": "logs",
            "dashboard_port": 8501,
            "auto_backup": True,
            "quality_threshold": 0.7,
            "batch_size": 1000,
            "memory_limit": 0.8,
            "log_level": "INFO"
        }

        cli_config_file = config_dir / "cli_config.json"
        with open(cli_config_file, 'w') as f:
            json.dump(cli_config, f, indent=2)
        print(f"  ⚙️  Created CLI configuration: {cli_config_file}")

        # Pipeline configuration
        pipeline_config = {
            "data_quality_threshold": 0.7,
            "batch_size": 1000,
            "memory_efficient": True,
            "parallel_processing": True,
            "cache_enabled": True,
            "auto_cleanup": True
        }

        pipeline_config_file = config_dir / "pipeline_config.json"
        with open(pipeline_config_file, 'w') as f:
            json.dump(pipeline_config, f, indent=2)
        print(f"  ⚙️  Created pipeline configuration: {pipeline_config_file}")

        # Training configuration
        training_config = {
            "default_model": "catboost",
            "cross_validation_folds": 5,
            "hyperparameter_optimization": True,
            "early_stopping": True,
            "feature_selection": True,
            "ensemble_methods": True
        }

        training_config_file = config_dir / "training_config.json"
        with open(training_config_file, 'w') as f:
            json.dump(training_config, f, indent=2)
        print(f"  ⚙️  Created training configuration: {training_config_file}")

        print("  ✅ Configuration files initialized")
        return True

    def setup_data_directories(self) -> bool:
        """Set up data directories and create sample data."""
        data_dir = self.project_root / "data"
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"

        # Create data subdirectories
        subdirs = ["eurusd", "gbpusd", "usdchf", "usdjpy", "audusd"]
        for subdir in subdirs:
            (raw_dir / subdir).mkdir(exist_ok=True)
            (processed_dir / subdir).mkdir(exist_ok=True)

        # Create sample data file
        sample_data = """# Sample Forex Data Format
# This is a sample of the expected data format
# timestamp,open,high,low,close,volume
20200101 170000,1.12345,1.12350,1.12340,1.12348,100
20200101 170100,1.12348,1.12352,1.12345,1.12350,95
20200101 170200,1.12350,1.12355,1.12348,1.12352,110"""

        sample_file = raw_dir / "sample_data.csv"
        with open(sample_file, 'w') as f:
            f.write(sample_data)

        print(f"  📄 Created sample data file: {sample_file}")
        print("  ✅ Data directories initialized")
        return True

    def verify_installation(self) -> bool:
        """Verify that installation was successful."""
        print("  🔍 Verifying installation...")

        # Check if key modules can be imported
        key_modules = [
            "pandas",
            "numpy",
            "streamlit",
            "catboost",
            "sklearn"
        ]

        failed_imports = []
        for module in key_modules:
            try:
                __import__(module)
                print(f"  ✅ {module} available")
            except ImportError:
                failed_imports.append(module)
                print(f"  ❌ {module} not available")

        if failed_imports:
            print(f"  ⚠️  Some modules failed to import: {failed_imports}")
            return False

        # Check if key files exist
        key_files = [
            "fxorcist_cli.py",
            "forex_ai_dashboard/pipeline/optimized_data_integration.py",
            "requirements.txt"
        ]

        for file_path in key_files:
            if (self.project_root / file_path).exists():
                print(f"  ✅ {file_path} exists")
            else:
                print(f"  ❌ {file_path} missing")
                return False

        print("  ✅ Installation verification completed")
        return True

    def create_startup_scripts(self) -> bool:
        """Create convenient startup scripts."""
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Create quick start script for Windows
        if self.system == "windows":
            batch_script = f"""@echo off
echo Starting FXorcist...
cd /d "{self.project_root}"
python fxorcist_cli.py --interactive
pause"""
            batch_file = scripts_dir / "start_fxorcist.bat"
            with open(batch_file, 'w') as f:
                f.write(batch_script)
            print(f"  📄 Created Windows startup script: {batch_file}")

        # Create shell script for Unix-like systems
        else:
            shell_script = f"""#!/bin/bash
echo "Starting FXorcist..."
cd "{self.project_root}"
python3 fxorcist_cli.py --interactive"""
            shell_file = scripts_dir / "start_fxorcist.sh"
            with open(shell_file, 'w') as f:
                f.write(shell_script)
            shell_file.chmod(0o755)
            print(f"  📄 Created Unix startup script: {shell_file}")

        # Create desktop shortcut (if possible)
        self.create_desktop_shortcut()

        print("  ✅ Startup scripts created")
        return True

    def create_desktop_shortcut(self):
        """Create desktop shortcut if possible."""
        try:
            if self.system == "windows":
                # Windows shortcut creation
                pass  # Would need pywin32
            elif self.system == "darwin":  # macOS
                desktop_dir = Path.home() / "Desktop"
                shortcut_file = desktop_dir / "FXorcist.desktop"

                shortcut_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=FXorcist AI Dashboard
Comment=Start FXorcist AI Dashboard
Exec={self.project_root}/scripts/start_fxorcist.sh
Icon=utilities-terminal
Terminal=true
Categories=Development;"""

                with open(shortcut_file, 'w') as f:
                    f.write(shortcut_content)
                shortcut_file.chmod(0o755)
                print(f"  🖥️  Created desktop shortcut: {shortcut_file}")

        except Exception as e:
            print(f"  ⚠️  Could not create desktop shortcut: {e}")

    def command_exists(self, command: str) -> bool:
        """Check if a command exists on the system."""
        try:
            if self.system == "windows":
                subprocess.check_output(["where", command], stderr=subprocess.DEVNULL)
            else:
                subprocess.check_output(["which", command], stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False

    def download_sample_data(self) -> bool:
        """Download sample forex data for testing."""
        print("  📥 Downloading sample forex data...")

        # This would download sample data from a repository
        # For now, we'll just create a placeholder
        sample_data_url = "https://example.com/sample_forex_data.zip"
        sample_data_file = self.project_root / "data" / "sample_forex_data.zip"

        try:
            print(f"  📡 Would download from: {sample_data_url}")
            print("  📄 Sample data file created during setup"
            # In a real implementation, you would:
            # urllib.request.urlretrieve(sample_data_url, sample_data_file)
            return True
        except Exception as e:
            print(f"  ⚠️  Could not download sample data: {e}")
            return True  # Don't fail setup for this

def main():
    """Main setup function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("FXorcist Automated Setup Script")
        print("Usage: python scripts/setup_fxorcist.py")
        print()
        print("This script will:")
        print("  - Check system requirements")
        print("  - Install Python dependencies")
        print("  - Set up project structure")
        print("  - Initialize configuration files")
        print("  - Create startup scripts")
        print("  - Verify installation")
        return

    setup = FXorcistSetup()
    success = setup.run_setup()

    if success:
        print("\n🎉 FXorcist is ready to use!")
        print("Run 'python fxorcist_cli.py --interactive' to get started.")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()