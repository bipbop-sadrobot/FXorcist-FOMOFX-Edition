"""
Tests for repository reorganization script.
Ensures safe and correct file organization.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
from scripts.reorganize_repo import RepoOrganizer

@pytest.fixture
def temp_repo():
    """Create a temporary repository structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        repo_dir = Path(temp_dir)
        
        # Create some test files
        (repo_dir / "comprehensive_2024_training.py").write_text("# Training code")
        (repo_dir / "comprehensive_2024_training_enhanced.py").write_text("# Enhanced training")
        (repo_dir / "advanced_system_dashboard.py").write_text("# Dashboard code")
        (repo_dir / "data_cleaning_fix.py").write_text("# Data cleaning")
        (repo_dir / "test_core.py").write_text("# Test file")
        (repo_dir / "federated.txt").write_text("Integration notes")
        
        # Create some nested files
        docs_dir = repo_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "GUIDE.md").write_text("Documentation")
        
        yield repo_dir

def test_create_package_structure(temp_repo):
    """Test creation of package directories."""
    organizer = RepoOrganizer(str(temp_repo))
    organizer.create_package_structure()
    
    # Verify all package directories were created
    expected_dirs = ['core', 'ui', 'tests', 'data', 'integrations']
    for dir_name in expected_dirs:
        dir_path = temp_repo / dir_name
        assert dir_path.exists()
        assert dir_path.is_dir()
        assert (dir_path / '__init__.py').exists()

def test_determine_package(temp_repo):
    """Test package determination for different files."""
    organizer = RepoOrganizer(str(temp_repo))
    
    # Test training file -> core
    assert organizer.determine_package(
        temp_repo / "comprehensive_2024_training.py"
    ) == "core"
    
    # Test dashboard file -> ui
    assert organizer.determine_package(
        temp_repo / "advanced_system_dashboard.py"
    ) == "ui"
    
    # Test data file -> data
    assert organizer.determine_package(
        temp_repo / "data_cleaning_fix.py"
    ) == "data"

def test_is_duplicate_training_file(temp_repo):
    """Test detection of duplicate training files."""
    organizer = RepoOrganizer(str(temp_repo))
    
    # Should detect training file variants
    assert organizer.is_duplicate_training_file(
        temp_repo / "comprehensive_2024_training_enhanced.py"
    )
    
    # Should not detect non-training files
    assert not organizer.is_duplicate_training_file(
        temp_repo / "advanced_system_dashboard.py"
    )

def test_normalize_filename(temp_repo):
    """Test filename normalization."""
    organizer = RepoOrganizer(str(temp_repo))
    
    assert organizer.normalize_filename("ComprehensiveTraining.py") == "comprehensive_training.py"
    assert organizer.normalize_filename("data-cleaning.py") == "data_cleaning.py"
    assert organizer.normalize_filename("TEST_FILE.py") == "test_file.py"

def test_organize_files(temp_repo):
    """Test complete file organization process."""
    organizer = RepoOrganizer(str(temp_repo))
    organizer.create_package_structure()
    organizer.organize_files()
    
    # Verify files were moved to correct packages
    assert (temp_repo / "core" / "comprehensive_2024_training.py").exists()
    assert (temp_repo / "ui" / "advanced_system_dashboard.py").exists()
    assert (temp_repo / "data" / "data_cleaning_fix.py").exists()
    
    # Verify duplicate was removed
    assert not (temp_repo / "comprehensive_2024_training_enhanced.py").exists()
    
    # Verify docs weren't moved (should stay in docs directory)
    assert (temp_repo / "docs" / "GUIDE.md").exists()

def test_safe_move_with_imports(temp_repo):
    """Test handling of files with dependencies."""
    # Create files with imports
    main_file = temp_repo / "main.py"
    main_file.write_text("""
from utils import helper
from .local_module import func
import numpy as np

def main():
    helper.do_something()
    """)
    
    utils_dir = temp_repo / "utils"
    utils_dir.mkdir()
    (utils_dir / "__init__.py").touch()
    (utils_dir / "helper.py").write_text("""
def do_something():
    pass
    """)
    
    organizer = RepoOrganizer(str(temp_repo))
    organizer.create_package_structure()
    organizer.organize_files()
    
    # Verify files were moved correctly
    assert (temp_repo / "core" / "main.py").exists()
    
    # Verify import-related files weren't moved
    assert (temp_repo / "utils" / "helper.py").exists()
    assert (temp_repo / "utils" / "__init__.py").exists()

def test_error_handling(temp_repo):
    """Test error handling during organization."""
    organizer = RepoOrganizer(str(temp_repo))
    
    # Test handling of non-existent file
    non_existent = temp_repo / "not_here.py"
    assert not organizer.should_delete_file(non_existent)
    
    # Test handling of directory instead of file
    dir_path = temp_repo / "test_dir"
    dir_path.mkdir()
    assert not organizer.should_delete_file(dir_path)

def test_no_circular_moves(temp_repo):
    """Test prevention of circular moves."""
    # Create a circular reference scenario
    (temp_repo / "a.py").write_text("from b import B")
    (temp_repo / "b.py").write_text("from a import A")
    
    organizer = RepoOrganizer(str(temp_repo))
    organizer.create_package_structure()
    organizer.organize_files()
    
    # Both files should end up in the same package
    assert (temp_repo / "core" / "a.py").exists()
    assert (temp_repo / "core" / "b.py").exists()

def test_consolidate_training_pipelines(temp_repo):
    """Test consolidation of training pipeline files."""
    # Create multiple training files
    files = [
        "training_pipeline.py",
        "training_pipeline_enhanced.py",
        "training_pipeline_fixed.py"
    ]
    for f in files:
        (temp_repo / f).write_text(f"# {f}")
    
    organizer = RepoOrganizer(str(temp_repo))
    organizer.create_package_structure()
    organizer.consolidate_training_pipelines()
    
    # Verify unified structure
    training_dir = temp_repo / "core" / "training"
    assert training_dir.exists()
    assert (training_dir / "base.py").exists()
    assert (training_dir / "enhanced.py").exists()
    
    # Original files should be moved or deleted
    for f in files:
        assert not (temp_repo / f).exists()