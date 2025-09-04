#!/usr/bin/env python3
"""
Repository reorganization script for FXorcist-FOMOFX-Edition.
Organizes files into logical packages and removes redundancy.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set
import re

class RepoOrganizer:
    """Handles repository reorganization and cleanup."""

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.packages = {
            'core': ['training', 'pipeline', 'model', 'feature', 'data_ingestion'],
            'ui': ['dashboard', 'cli', 'visualization'],
            'tests': ['test_'],
            'data': ['dataset', 'validation', 'ingestion'],
            'integrations': ['integration', 'memory_system', 'federated']
        }
        self.moved_files: Dict[str, str] = {}
        self.deleted_files: Set[str] = set()

    def create_package_structure(self) -> None:
        """Create the main package directories."""
        for pkg in self.packages:
            pkg_dir = self.root / pkg
            pkg_dir.mkdir(exist_ok=True)
            (pkg_dir / '__init__.py').touch()
            print(f"Created package directory: {pkg}")

    def is_duplicate_training_file(self, file: Path) -> bool:
        """Check if a file is a duplicate training pipeline."""
        if not file.name.endswith('.py'):
            return False
        
        training_patterns = [
            r'.*training_pipeline.*\.py$',
            r'.*train.*enhanced.*\.py$',
            r'.*comprehensive.*training.*\.py$'
        ]
        
        return any(re.match(pattern, file.name) for pattern in training_patterns)

    def determine_package(self, file: Path) -> str:
        """Determine which package a file belongs to."""
        file_content = file.read_text() if file.is_file() else ""
        
        # Check file name and content against package patterns
        for pkg, patterns in self.packages.items():
            if any(pattern in file.name.lower() for pattern in patterns):
                return pkg
            if any(pattern in file_content.lower() for pattern in patterns):
                return pkg
        
        return 'core'  # Default to core package

    def should_delete_file(self, file: Path) -> bool:
        """Determine if a file should be deleted."""
        if not file.is_file():
            return False

        # Check for duplicate training files
        if self.is_duplicate_training_file(file):
            similar_files = [f for f in self.moved_files if 'training' in f.lower()]
            if similar_files:
                return True

        # Check for obsolete or backup files
        obsolete_patterns = [
            r'.*\.bak$',
            r'.*\.old$',
            r'.*_backup\..*$',
            r'.*_deprecated\..*$'
        ]
        return any(re.match(pattern, file.name) for pattern in obsolete_patterns)

    def normalize_filename(self, filename: str) -> str:
        """Normalize filename to lowercase with underscores."""
        # Remove special characters and convert to lowercase
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', filename.lower())
        # Convert camelCase to snake_case
        normalized = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', normalized).lower()
        # Clean up multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        return normalized

    def organize_files(self) -> None:
        """Organize files into appropriate packages."""
        print("Starting file organization...")
        
        # First pass: Identify and mark files for moving/deletion
        for file in self.root.rglob('*'):
            if file.is_file() and not any(p in str(file) for p in self.packages.keys()):
                if self.should_delete_file(file):
                    self.deleted_files.add(str(file))
                    continue

                pkg = self.determine_package(file)
                if pkg:
                    normalized_name = self.normalize_filename(file.name)
                    new_path = self.root / pkg / normalized_name
                    self.moved_files[str(file)] = str(new_path)

        # Second pass: Move files
        for old_path, new_path in self.moved_files.items():
            old_path = Path(old_path)
            new_path = Path(new_path)
            
            if old_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_path), str(new_path))
                print(f"Moved: {old_path} -> {new_path}")

        # Third pass: Delete marked files
        for file in self.deleted_files:
            if Path(file).exists():
                os.remove(file)
                print(f"Deleted: {file}")

    def consolidate_training_pipelines(self) -> None:
        """Consolidate multiple training pipeline files into a unified structure."""
        training_files = [
            f for f in self.root.rglob('*training*.py')
            if 'test' not in f.name.lower()
        ]

        if not training_files:
            return

        # Create unified training module
        unified_dir = self.root / 'core' / 'training'
        unified_dir.mkdir(exist_ok=True)

        # Create specialized modules
        modules = {
            'base.py': 'Base training pipeline implementation',
            'enhanced.py': 'Enhanced training with advanced features',
            'federated.py': 'Federated learning implementation',
            'memory.py': 'Memory-based training components',
            'utils.py': 'Training utilities and helpers'
        }

        for module, description in modules.items():
            module_path = unified_dir / module
            if not module_path.exists():
                module_path.write_text(f'"""\n{description}\n"""\n\n')

        print("Created unified training structure in core/training/")

    def run(self) -> None:
        """Execute the complete reorganization process."""
        print(f"Starting repository reorganization for: {self.root}")
        
        # Create package structure
        self.create_package_structure()
        
        # Organize files into packages
        self.organize_files()
        
        # Consolidate training pipelines
        self.consolidate_training_pipelines()
        
        print("\nReorganization complete!")
        print(f"Moved files: {len(self.moved_files)}")
        print(f"Deleted files: {len(self.deleted_files)}")
        print("\nPlease review the changes and update imports as needed.")

def main():
    """Main entry point."""
    import sys
    if len(sys.argv) != 2:
        print("Usage: python reorganize_repo.py /path/to/repo")
        sys.exit(1)

    organizer = RepoOrganizer(sys.argv[1])
    organizer.run()

if __name__ == "__main__":
    main()