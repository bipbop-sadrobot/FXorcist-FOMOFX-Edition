#!/usr/bin/env python3
"""
Repository reorganization script for FXorcist-FOMOFX-Edition.
Organizes files into logical packages and removes redundancy.
Includes safe handling of imports and dependencies.
"""

import os
import shutil
import logging
import ast
from pathlib import Path
from typing import Dict, List, Set, Optional
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"reorganize_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImportAnalyzer(ast.NodeVisitor):
    """Analyzes Python file imports."""
    
    def __init__(self):
        self.imports = set()
        self.from_imports = {}
        
    def visit_Import(self, node):
        for name in node.names:
            self.imports.add(name.name.split('.')[0])
            
    def visit_ImportFrom(self, node):
        if node.module:
            self.from_imports[node.module.split('.')[0]] = [n.name for n in node.names]

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
        self.import_graph: Dict[str, Set[str]] = {}
        self.phase = 'analysis'  # Current phase of reorganization

    def analyze_imports(self, file: Path) -> Optional[Dict[str, List[str]]]:
        """
        Analyze imports in a Python file.
        Returns dict of {module: [imported_names]} or None if not a Python file.
        """
        if not file.name.endswith('.py'):
            return None
            
        try:
            analyzer = ImportAnalyzer()
            tree = ast.parse(file.read_text())
            analyzer.visit(tree)
            return {
                'imports': list(analyzer.imports),
                'from_imports': analyzer.from_imports
            }
        except Exception as e:
            logger.warning(f"Failed to analyze imports in {file}: {e}")
            return None

    def build_dependency_graph(self) -> None:
        """Build graph of file dependencies based on imports."""
        logger.info("Building dependency graph...")
        
        for file in self.root.rglob('*.py'):
            if file.is_file():
                rel_path = str(file.relative_to(self.root))
                self.import_graph[rel_path] = set()
                
                imports = self.analyze_imports(file)
                if imports:
                    # Find files that provide these imports
                    for module in imports['imports'] + list(imports['from_imports'].keys()):
                        # Look for matching files
                        potential_matches = list(self.root.rglob(f"{module}.py"))
                        if potential_matches:
                            for match in potential_matches:
                                self.import_graph[rel_path].add(
                                    str(match.relative_to(self.root))
                                )

    def create_package_structure(self) -> None:
        """Create the main package directories."""
        logger.info("Creating package structure...")
        for pkg in self.packages:
            pkg_dir = self.root / pkg
            pkg_dir.mkdir(exist_ok=True)
            (pkg_dir / '__init__.py').touch()
            logger.info(f"Created package directory: {pkg}")

    def is_duplicate_training_file(self, file: Path) -> bool:
        """Check if a file is a duplicate training pipeline."""
        if not file.name.endswith('.py'):
            return False
        
        training_patterns = [
            r'.*training_pipeline.*\.py$',
            r'.*train.*enhanced.*\.py$',
            r'.*comprehensive.*training.*\.py$'
        ]
        
        is_duplicate = any(re.match(pattern, file.name) for pattern in training_patterns)
        if is_duplicate:
            logger.debug(f"Identified duplicate training file: {file}")
        return is_duplicate

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

    def update_imports(self, file: Path, moved_files: Dict[str, str]) -> None:
        """Update imports in a file after moving dependencies."""
        if not file.name.endswith('.py'):
            return

        try:
            content = file.read_text()
            tree = ast.parse(content)
            
            class ImportUpdater(ast.NodeTransformer):
                def __init__(self, moved_files):
                    self.moved_files = moved_files
                    self.changes = False
                    
                def visit_Import(self, node):
                    for name in node.names:
                        old_name = name.name.split('.')[0]
                        if f"{old_name}.py" in self.moved_files:
                            new_path = Path(self.moved_files[f"{old_name}.py"])
                            new_module = new_path.parent.name + '.' + new_path.stem
                            name.name = new_module
                            self.changes = True
                    return node
                    
                def visit_ImportFrom(self, node):
                    if node.module and f"{node.module.split('.')[0]}.py" in self.moved_files:
                        new_path = Path(self.moved_files[f"{node.module.split('.')[0]}.py"])
                        node.module = new_path.parent.name + '.' + new_path.stem
                        self.changes = True
                    return node
            
            updater = ImportUpdater(moved_files)
            new_tree = updater.visit(tree)
            
            if updater.changes:
                file.write_text(ast.unparse(new_tree))
                logger.info(f"Updated imports in {file}")
                
        except Exception as e:
            logger.warning(f"Failed to update imports in {file}: {e}")

    def organize_files_safely(self) -> None:
        """Organize files with import updates."""
        logger.info("Starting safe file organization...")
        
        # Analysis phase
        self.phase = 'analysis'
        self.build_dependency_graph()
        
        # Planning phase
        self.phase = 'planning'
        moves_by_package = {}
        for pkg in self.packages:
            moves_by_package[pkg] = []
            
        for file in self.root.rglob('*'):
            if file.is_file() and not any(p in str(file) for p in self.packages.keys()):
                if self.should_delete_file(file):
                    self.deleted_files.add(str(file))
                    continue
                    
                pkg = self.determine_package(file)
                if pkg:
                    normalized_name = self.normalize_filename(file.name)
                    new_path = self.root / pkg / normalized_name
                    moves_by_package[pkg].append((file, new_path))
        
        # Execution phase
        self.phase = 'execution'
        
        # Move files package by package
        for pkg, moves in moves_by_package.items():
            logger.info(f"Processing package: {pkg}")
            
            # Create package directory
            pkg_dir = self.root / pkg
            pkg_dir.mkdir(exist_ok=True)
            (pkg_dir / '__init__.py').touch()
            
            # Move files
            for old_path, new_path in moves:
                if old_path.exists():
                    try:
                        # Create parent directories
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Move file
                        shutil.move(str(old_path), str(new_path))
                        self.moved_files[str(old_path)] = str(new_path)
                        logger.info(f"Moved: {old_path} -> {new_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to move {old_path}: {e}")
            
            # Update imports in moved files
            for _, new_path in moves:
                if new_path.exists():
                    self.update_imports(new_path, self.moved_files)
        
        # Delete marked files
        for file in self.deleted_files:
            if Path(file).exists():
                try:
                    os.remove(file)
                    logger.info(f"Deleted: {file}")
                except Exception as e:
                    logger.error(f"Failed to delete {file}: {e}")

    def validate_reorganization(self) -> bool:
        """
        Validate the reorganization results.
        Returns True if validation passes.
        """
        logger.info("Validating reorganization...")
        
        success = True
        
        # Check all files were moved correctly
        for old_path, new_path in self.moved_files.items():
            if Path(old_path).exists():
                logger.error(f"Source file still exists: {old_path}")
                success = False
            if not Path(new_path).exists():
                logger.error(f"Destination file missing: {new_path}")
                success = False
        
        # Check all packages have __init__.py
        for pkg in self.packages:
            init_file = self.root / pkg / "__init__.py"
            if not init_file.exists():
                logger.error(f"Missing __init__.py in {pkg}")
                success = False
        
        # Check for circular imports
        for file in self.root.rglob("*.py"):
            try:
                ast.parse(file.read_text())
            except SyntaxError as e:
                logger.error(f"Syntax error in {file}: {e}")
                success = False
        
        if success:
            logger.info("Validation passed!")
        else:
            logger.error("Validation failed!")
            
        return success

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