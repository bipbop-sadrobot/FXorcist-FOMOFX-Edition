"""
Configuration migration system.

Handles configuration version migrations and rollbacks.
"""

import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
import yaml
from datetime import datetime
import shutil

from .config import FXorcistConfig

# Setup logging
logger = logging.getLogger(__name__)

class MigrationError(Exception):
    """Base exception for migration errors."""
    pass

class Migration:
    """Configuration migration definition."""
    
    def __init__(
        self,
        from_version: str,
        to_version: str,
        description: str,
        migration_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """Initialize migration.
        
        Args:
            from_version: Source version
            to_version: Target version
            description: Migration description
            migration_func: Migration function
        """
        self.from_version = from_version
        self.to_version = to_version
        self.description = description
        self.migration_func = migration_func
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert migration to dictionary.
        
        Returns:
            Migration data dictionary
        """
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "description": self.description,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Migration':
        """Create migration from dictionary.
        
        Args:
            data: Migration data dictionary
            
        Returns:
            Migration instance
        """
        return cls(
            from_version=data["from_version"],
            to_version=data["to_version"],
            description=data["description"],
            migration_func=lambda x: x  # Placeholder
        )

class MigrationManager:
    """Manages configuration migrations."""
    
    def __init__(self, migrations_dir: Optional[Path] = None):
        """Initialize migration manager.
        
        Args:
            migrations_dir: Optional migrations directory path
        """
        self.migrations_dir = migrations_dir or Path("config/migrations")
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.migrations: Dict[str, Migration] = {}
        self._load_migrations()
    
    def _load_migrations(self):
        """Load registered migrations."""
        migrations_file = self.migrations_dir / "migrations.json"
        if migrations_file.exists():
            with open(migrations_file, 'r') as f:
                data = json.load(f)
                for key, migration_data in data.items():
                    self.migrations[key] = Migration.from_dict(migration_data)
    
    def _save_migrations(self):
        """Save registered migrations."""
        migrations_file = self.migrations_dir / "migrations.json"
        migrations_data = {
            key: migration.to_dict()
            for key, migration in self.migrations.items()
        }
        with open(migrations_file, 'w') as f:
            json.dump(migrations_data, f, indent=2)
    
    def register_migration(self, migration: Migration):
        """Register new migration.
        
        Args:
            migration: Migration to register
        """
        key = f"{migration.from_version}_to_{migration.to_version}"
        self.migrations[key] = migration
        self._save_migrations()
        logger.info(f"Registered migration: {key}")
    
    def create_backup(self, config: FXorcistConfig) -> str:
        """Create configuration backup.
        
        Args:
            config: Configuration to backup
            
        Returns:
            Backup filename
        """
        backup_dir = self.migrations_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        # Create backup filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"config_v{config.version}_{timestamp}.yaml"
        
        # Save backup
        with open(backup_file, 'w') as f:
            yaml.dump(config.dict(), f, default_flow_style=False)
        
        logger.info(f"Created backup: {backup_file}")
        return backup_file.name
    
    def restore_backup(self, backup_file: str) -> Dict[str, Any]:
        """Restore configuration from backup.
        
        Args:
            backup_file: Backup filename
            
        Returns:
            Restored configuration dictionary
        """
        backup_path = self.migrations_dir / "backups" / backup_file
        if not backup_path.exists():
            raise MigrationError(f"Backup file not found: {backup_file}")
        
        with open(backup_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        logger.info(f"Restored backup: {backup_file}")
        return config_data
    
    def get_migration_path(
        self,
        from_version: str,
        to_version: str
    ) -> List[Migration]:
        """Get migration path between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            
        Returns:
            List of migrations to apply
        """
        if from_version == to_version:
            return []
        
        # Find direct migration
        direct_key = f"{from_version}_to_{to_version}"
        if direct_key in self.migrations:
            return [self.migrations[direct_key]]
        
        # Find migration path
        path = []
        current = from_version
        
        while current != to_version:
            next_migration = None
            for key, migration in self.migrations.items():
                if migration.from_version == current:
                    if migration.to_version == to_version:
                        path.append(migration)
                        return path
                    elif not next_migration:
                        next_migration = migration
            
            if not next_migration:
                raise MigrationError(
                    f"No migration path from {from_version} to {to_version}"
                )
            
            path.append(next_migration)
            current = next_migration.to_version
        
        return path
    
    def migrate(
        self,
        config_data: Dict[str, Any],
        target_version: str
    ) -> Dict[str, Any]:
        """Migrate configuration to target version.
        
        Args:
            config_data: Configuration dictionary
            target_version: Target version
            
        Returns:
            Migrated configuration dictionary
        """
        current_version = config_data.get("version", "1.0.0")
        
        # Get migration path
        try:
            migration_path = self.get_migration_path(
                current_version,
                target_version
            )
        except MigrationError as e:
            logger.error(f"Migration path error: {e}")
            raise
        
        # Apply migrations
        for migration in migration_path:
            try:
                # Create backup
                config = FXorcistConfig(**config_data)
                self.create_backup(config)
                
                # Apply migration
                config_data = migration.migration_func(config_data)
                config_data["version"] = migration.to_version
                
                logger.info(
                    f"Applied migration: {migration.from_version} -> "
                    f"{migration.to_version}"
                )
                
            except Exception as e:
                logger.error(f"Migration failed: {e}")
                raise MigrationError(f"Migration failed: {e}")
        
        return config_data
    
    def rollback(
        self,
        config_data: Dict[str, Any],
        target_version: str
    ) -> Dict[str, Any]:
        """Rollback configuration to target version.
        
        Args:
            config_data: Configuration dictionary
            target_version: Target version
            
        Returns:
            Rolled back configuration dictionary
        """
        current_version = config_data.get("version", "1.0.0")
        
        # Find backup file
        backup_dir = self.migrations_dir / "backups"
        backup_files = sorted(
            backup_dir.glob(f"config_v{target_version}_*.yaml"),
            reverse=True
        )
        
        if not backup_files:
            raise MigrationError(
                f"No backup found for version {target_version}"
            )
        
        # Restore backup
        backup_data = self.restore_backup(backup_files[0].name)
        
        logger.info(f"Rolled back from {current_version} to {target_version}")
        return backup_data

# Example migrations
def migrate_1_0_0_to_1_1_0(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from v1.0.0 to v1.1.0."""
    result = config.copy()
    
    # Add new fields
    if 'risk' in result:
        result['risk']['max_drawdown_limit'] = 0.10
    
    return result

def migrate_1_1_0_to_1_2_0(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from v1.1.0 to v1.2.0."""
    result = config.copy()
    
    # Update trading hours format
    if 'trading' in result and 'trading_hours' in result['trading']:
        hours = result['trading']['trading_hours']
        result['trading']['trading_hours'] = {
            'start': hours.get('start', '00:00'),
            'end': hours.get('end', '23:59'),
            'timezone': 'UTC'
        }
    
    return result

# Register example migrations
migration_manager = MigrationManager()
migration_manager.register_migration(
    Migration(
        "1.0.0",
        "1.1.0",
        "Add max drawdown limit",
        migrate_1_0_0_to_1_1_0
    )
)
migration_manager.register_migration(
    Migration(
        "1.1.0",
        "1.2.0",
        "Update trading hours format",
        migrate_1_1_0_to_1_2_0
    )
)