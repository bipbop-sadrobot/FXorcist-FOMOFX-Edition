#!/usr/bin/env python3
"""
Advanced Configuration Management System for FXorcist

This module provides comprehensive configuration management with:
- Environment-specific configurations
- Configuration validation and schema enforcement
- Dynamic configuration updates
- Integration with CLI and dashboard systems
- Configuration migration and versioning
- Encrypted sensitive data handling
- Real-time configuration synchronization

Author: FXorcist Development Team
Version: 2.0
Date: September 2, 2025
"""

import os
import json
import yaml
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
from functools import wraps
import jsonschema
from cryptography.fernet import Fernet
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    name: str
    version: str
    schema: Dict[str, Any]
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Callable] = field(default_factory=dict)

@dataclass
class ConfigInstance:
    """Configuration instance with metadata"""
    environment: str
    data: Dict[str, Any]
    schema_version: str
    created_at: datetime
    modified_at: datetime
    checksum: str
    encrypted_fields: List[str] = field(default_factory=list)

class ConfigurationManager:
    """Advanced configuration management system"""

    def __init__(self, config_dir: str = "config", encryption_key: Optional[str] = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Encryption setup
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.cipher = Fernet(self.encryption_key)

        # Configuration storage
        self.schemas: Dict[str, ConfigSchema] = {}
        self.configs: Dict[str, ConfigInstance] = {}
        self.listeners: Dict[str, List[Callable]] = {}

        # Threading
        self.lock = threading.RLock()
        self.monitor_thread = None
        self.stop_monitoring = False

        # Load default schemas
        self._load_default_schemas()

        # Load existing configurations
        self._load_configurations()

        # Start file monitoring
        self.start_file_monitoring()

    def _generate_encryption_key(self) -> str:
        """Generate encryption key from environment or create new one"""
        key_file = self.config_dir / ".encryption_key"

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            return key

    def _load_default_schemas(self):
        """Load default configuration schemas"""
        self.schemas['cli'] = ConfigSchema(
            name='cli',
            version='2.0',
            schema={
                'type': 'object',
                'properties': {
                    'data_dir': {'type': 'string'},
                    'models_dir': {'type': 'string'},
                    'logs_dir': {'type': 'string'},
                    'dashboard_port': {'type': 'integer', 'minimum': 1024, 'maximum': 65535},
                    'auto_backup': {'type': 'boolean'},
                    'quality_threshold': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                    'batch_size': {'type': 'integer', 'minimum': 1},
                    'log_level': {'type': 'string', 'enum': ['DEBUG', 'INFO', 'WARNING', 'ERROR']},
                    'max_memory_usage': {'type': 'number', 'minimum': 0.1, 'maximum': 1.0},
                    'parallel_processing': {'type': 'boolean'},
                    'cache_enabled': {'type': 'boolean'}
                },
                'required': ['data_dir', 'models_dir', 'logs_dir']
            },
            required_fields=['data_dir', 'models_dir', 'logs_dir'],
            validation_rules={
                'dashboard_port': lambda x: 1024 <= x <= 65535,
                'quality_threshold': lambda x: 0.0 <= x <= 1.0,
                'batch_size': lambda x: x > 0
            }
        )

        self.schemas['training'] = ConfigSchema(
            name='training',
            version='2.0',
            schema={
                'type': 'object',
                'properties': {
                    'default_model': {'type': 'string', 'enum': ['catboost', 'lightgbm', 'xgboost', 'random_forest']},
                    'cross_validation_folds': {'type': 'integer', 'minimum': 2, 'maximum': 10},
                    'hyperparameter_optimization': {'type': 'boolean'},
                    'n_trials': {'type': 'integer', 'minimum': 10, 'maximum': 1000},
                    'early_stopping': {'type': 'boolean'},
                    'feature_selection': {'type': 'boolean'},
                    'ensemble_methods': {'type': 'boolean'},
                    'max_training_time': {'type': 'integer', 'minimum': 60},
                    'validation_split': {'type': 'number', 'minimum': 0.1, 'maximum': 0.5}
                },
                'required': ['default_model']
            }
        )

        self.schemas['dashboard'] = ConfigSchema(
            name='dashboard',
            version='2.0',
            schema={
                'type': 'object',
                'properties': {
                    'theme': {'type': 'string', 'enum': ['light', 'dark', 'auto']},
                    'auto_refresh': {'type': 'boolean'},
                    'refresh_interval': {'type': 'integer', 'minimum': 5, 'maximum': 300},
                    'max_chart_points': {'type': 'integer', 'minimum': 100},
                    'enable_caching': {'type': 'boolean'},
                    'cache_ttl': {'type': 'integer', 'minimum': 60},
                    'export_formats': {'type': 'array', 'items': {'type': 'string'}},
                    'default_timeframe': {'type': 'string'}
                }
            }
        )

    def _load_configurations(self):
        """Load existing configuration files"""
        for config_file in self.config_dir.glob("*.json"):
            if config_file.name.startswith('.'):
                continue

            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)

                if 'schema_version' in data:
                    # New format with metadata
                    instance = ConfigInstance(
                        environment=config_file.stem,
                        data=data['data'],
                        schema_version=data['schema_version'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        modified_at=datetime.fromisoformat(data['modified_at']),
                        checksum=data['checksum'],
                        encrypted_fields=data.get('encrypted_fields', [])
                    )

                    # Decrypt sensitive fields
                    self._decrypt_sensitive_fields(instance)
                    self.configs[config_file.stem] = instance
                else:
                    # Legacy format - migrate
                    self._migrate_legacy_config(config_file.stem, data)

            except Exception as e:
                logger.error(f"Error loading configuration {config_file}: {e}")

    def _migrate_legacy_config(self, environment: str, data: Dict[str, Any]):
        """Migrate legacy configuration format"""
        logger.info(f"Migrating legacy configuration for {environment}")

        instance = ConfigInstance(
            environment=environment,
            data=data,
            schema_version='1.0',
            created_at=datetime.now(),
            modified_at=datetime.now(),
            checksum=self._calculate_checksum(data),
            encrypted_fields=[]
        )

        self.configs[environment] = instance
        self.save_configuration(environment)

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate configuration checksum"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _encrypt_sensitive_data(self, value: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(value.encode()).decode()

    def _decrypt_sensitive_data(self, value: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(value.encode()).decode()

    def _encrypt_sensitive_fields(self, instance: ConfigInstance):
        """Encrypt sensitive fields in configuration"""
        sensitive_patterns = ['password', 'secret', 'key', 'token']

        for key, value in instance.data.items():
            if any(pattern in key.lower() for pattern in sensitive_patterns):
                if isinstance(value, str) and not value.startswith('encrypted:'):
                    instance.data[key] = f"encrypted:{self._encrypt_sensitive_data(value)}"
                    if key not in instance.encrypted_fields:
                        instance.encrypted_fields.append(key)

    def _decrypt_sensitive_fields(self, instance: ConfigInstance):
        """Decrypt sensitive fields in configuration"""
        for field in instance.encrypted_fields:
            if field in instance.data:
                value = instance.data[field]
                if isinstance(value, str) and value.startswith('encrypted:'):
                    try:
                        instance.data[field] = self._decrypt_sensitive_data(value[10:])  # Remove 'encrypted:' prefix
                    except Exception as e:
                        logger.error(f"Failed to decrypt field {field}: {e}")

    def create_configuration(self, environment: str, schema_name: str,
                           initial_data: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new configuration instance"""
        with self.lock:
            if environment in self.configs:
                logger.warning(f"Configuration {environment} already exists")
                return False

            if schema_name not in self.schemas:
                logger.error(f"Schema {schema_name} not found")
                return False

            schema = self.schemas[schema_name]
            data = initial_data or {}

            # Validate configuration
            if not self._validate_configuration(data, schema):
                return False

            # Create instance
            instance = ConfigInstance(
                environment=environment,
                data=data,
                schema_version=schema.version,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                checksum=self._calculate_checksum(data)
            )

            # Encrypt sensitive fields
            self._encrypt_sensitive_fields(instance)

            self.configs[environment] = instance
            self.save_configuration(environment)

            logger.info(f"Created configuration {environment} with schema {schema_name}")
            return True

    def get_configuration(self, environment: str, decrypt: bool = True) -> Optional[Dict[str, Any]]:
        """Get configuration for environment"""
        with self.lock:
            if environment not in self.configs:
                return None

            instance = self.configs[environment]

            # Return a copy to prevent external modifications
            data = instance.data.copy()

            if decrypt:
                # Create temporary instance for decryption
                temp_instance = ConfigInstance(
                    environment=instance.environment,
                    data=data,
                    schema_version=instance.schema_version,
                    created_at=instance.created_at,
                    modified_at=instance.modified_at,
                    checksum=instance.checksum,
                    encrypted_fields=instance.encrypted_fields.copy()
                )
                self._decrypt_sensitive_fields(temp_instance)
                return temp_instance.data

            return data

    def update_configuration(self, environment: str, updates: Dict[str, Any],
                           validate: bool = True) -> bool:
        """Update configuration with new values"""
        with self.lock:
            if environment not in self.configs:
                logger.error(f"Configuration {environment} not found")
                return False

            instance = self.configs[environment]
            schema = self.schemas.get(instance.schema_version.split('.')[0], self.schemas.get('cli'))

            # Apply updates
            self._deep_update(instance.data, updates)

            # Validate if requested
            if validate and not self._validate_configuration(instance.data, schema):
                return False

            # Update metadata
            instance.modified_at = datetime.now()
            instance.checksum = self._calculate_checksum(instance.data)

            # Encrypt sensitive fields
            self._encrypt_sensitive_fields(instance)

            # Save configuration
            self.save_configuration(environment)

            # Notify listeners
            self._notify_listeners(environment, instance.data)

            logger.info(f"Updated configuration {environment}")
            return True

    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]):
        """Deep update nested dictionary"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _validate_configuration(self, data: Dict[str, Any], schema: ConfigSchema) -> bool:
        """Validate configuration against schema"""
        try:
            # JSON Schema validation
            jsonschema.validate(data, schema.schema)

            # Custom validation rules
            for field, validator in schema.validation_rules.items():
                if field in data:
                    if not validator(data[field]):
                        logger.error(f"Validation failed for field {field}")
                        return False

            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Schema validation error: {e}")
            return False
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def save_configuration(self, environment: str):
        """Save configuration to file"""
        if environment not in self.configs:
            return

        instance = self.configs[environment]
        config_file = self.config_dir / f"{environment}.json"

        # Prepare data for saving
        save_data = {
            'data': instance.data,
            'schema_version': instance.schema_version,
            'created_at': instance.created_at.isoformat(),
            'modified_at': instance.modified_at.isoformat(),
            'checksum': instance.checksum,
            'encrypted_fields': instance.encrypted_fields
        }

        try:
            with open(config_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.debug(f"Saved configuration {environment}")
        except Exception as e:
            logger.error(f"Error saving configuration {environment}: {e}")

    def delete_configuration(self, environment: str) -> bool:
        """Delete configuration"""
        with self.lock:
            if environment not in self.configs:
                return False

            # Remove from memory
            del self.configs[environment]

            # Remove file
            config_file = self.config_dir / f"{environment}.json"
            if config_file.exists():
                config_file.unlink()

            logger.info(f"Deleted configuration {environment}")
            return True

    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all configurations with metadata"""
        with self.lock:
            return [{
                'environment': env,
                'schema_version': instance.schema_version,
                'created_at': instance.created_at.isoformat(),
                'modified_at': instance.modified_at.isoformat(),
                'checksum': instance.checksum
            } for env, instance in self.configs.items()]

    def add_change_listener(self, environment: str, callback: Callable):
        """Add configuration change listener"""
        if environment not in self.listeners:
            self.listeners[environment] = []
        self.listeners[environment].append(callback)

    def _notify_listeners(self, environment: str, new_data: Dict[str, Any]):
        """Notify listeners of configuration changes"""
        if environment in self.listeners:
            for callback in self.listeners[environment]:
                try:
                    callback(environment, new_data)
                except Exception as e:
                    logger.error(f"Error in configuration listener: {e}")

    def start_file_monitoring(self):
        """Start file system monitoring for configuration changes"""
        self.monitor_thread = threading.Thread(target=self._monitor_files, daemon=True)
        self.monitor_thread.start()

    def _monitor_files(self):
        """Monitor configuration files for external changes"""
        last_checksums = {}

        while not self.stop_monitoring:
            try:
                for config_file in self.config_dir.glob("*.json"):
                    if config_file.name.startswith('.'):
                        continue

                    current_checksum = self._file_checksum(config_file)

                    if str(config_file) in last_checksums:
                        if last_checksums[str(config_file)] != current_checksum:
                            logger.info(f"Configuration file {config_file.name} changed externally")
                            # Reload configuration
                            environment = config_file.stem
                            if environment in self.configs:
                                del self.configs[environment]
                            self._load_configurations()

                    last_checksums[str(config_file)] = current_checksum

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in file monitoring: {e}")
                time.sleep(10)

    def _file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return ""

    def export_configuration(self, environment: str, format: str = 'json') -> Optional[str]:
        """Export configuration in specified format"""
        config = self.get_configuration(environment)
        if not config:
            return None

        if format == 'json':
            return json.dumps(config, indent=2)
        elif format == 'yaml':
            try:
                import yaml
                return yaml.dump(config, default_flow_style=False)
            except ImportError:
                logger.error("PyYAML not installed for YAML export")
                return None
        else:
            logger.error(f"Unsupported export format: {format}")
            return None

    def import_configuration(self, environment: str, data: str, format: str = 'json') -> bool:
        """Import configuration from string"""
        try:
            if format == 'json':
                config_data = json.loads(data)
            elif format == 'yaml':
                import yaml
                config_data = yaml.safe_load(data)
            else:
                logger.error(f"Unsupported import format: {format}")
                return False

            return self.update_configuration(environment, config_data, validate=True)
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False

    def get_configuration_template(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration template for schema"""
        if schema_name not in self.schemas:
            return None

        schema = self.schemas[schema_name]

        # Generate template from schema
        template = {}

        def generate_from_schema(schema_part: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            properties = schema_part.get('properties', {})

            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get('type')

                if prop_type == 'string':
                    if 'enum' in prop_schema:
                        result[prop_name] = prop_schema['enum'][0]
                    else:
                        result[prop_name] = f"your_{prop_name}"
                elif prop_type == 'integer':
                    minimum = prop_schema.get('minimum', 0)
                    result[prop_name] = minimum
                elif prop_type == 'number':
                    minimum = prop_schema.get('minimum', 0.0)
                    result[prop_name] = minimum
                elif prop_type == 'boolean':
                    result[prop_name] = True
                elif prop_type == 'array':
                    result[prop_name] = []
                elif prop_type == 'object':
                    result[prop_name] = generate_from_schema(prop_schema)

            return result

        return generate_from_schema(schema.schema)

    def validate_all_configurations(self) -> Dict[str, List[str]]:
        """Validate all configurations and return errors"""
        errors = {}

        for environment, instance in self.configs.items():
            schema_name = instance.schema_version.split('.')[0]
            schema = self.schemas.get(schema_name, self.schemas.get('cli'))

            if schema:
                try:
                    jsonschema.validate(instance.data, schema.schema)
                except jsonschema.ValidationError as e:
                    errors[environment] = [str(e)]
                except Exception as e:
                    errors[environment] = [f"Validation error: {e}"]
            else:
                errors[environment] = [f"Schema {schema_name} not found"]

        return errors

    def migrate_configuration(self, environment: str, target_schema: str) -> bool:
        """Migrate configuration to new schema version"""
        if environment not in self.configs:
            return False

        if target_schema not in self.schemas:
            return False

        instance = self.configs[environment]
        new_schema = self.schemas[target_schema]

        # Basic migration - in production, this would be more sophisticated
        instance.schema_version = new_schema.version
        instance.modified_at = datetime.now()

        # Validate with new schema
        if self._validate_configuration(instance.data, new_schema):
            self.save_configuration(environment)
            logger.info(f"Migrated {environment} to schema {target_schema}")
            return True
        else:
            logger.error(f"Migration failed for {environment}")
            return False

    def shutdown(self):
        """Shutdown configuration manager"""
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # Save all configurations
        for environment in self.configs:
            self.save_configuration(environment)

        logger.info("Configuration manager shutdown complete")

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def config_property(environment: str, key: str, default=None):
    """Decorator to create configuration-backed properties"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_config_manager()
            config = manager.get_configuration(environment)
            if config and key in config:
                return config[key]
            return default
        return wrapper
    return decorator

# CLI Integration functions
def load_cli_config() -> Dict[str, Any]:
    """Load CLI configuration"""
    manager = get_config_manager()
    config = manager.get_configuration('cli')

    if not config:
        # Create default CLI configuration
        default_config = {
            'data_dir': 'data',
            'models_dir': 'models',
            'logs_dir': 'logs',
            'dashboard_port': 8501,
            'auto_backup': True,
            'quality_threshold': 0.7,
            'batch_size': 1000,
            'log_level': 'INFO',
            'max_memory_usage': 0.8,
            'parallel_processing': True,
            'cache_enabled': True
        }

        manager.create_configuration('cli', 'cli', default_config)
        return default_config

    return config

def update_cli_config(updates: Dict[str, Any]) -> bool:
    """Update CLI configuration"""
    manager = get_config_manager()
    return manager.update_configuration('cli', updates)

if __name__ == "__main__":
    # Example usage
    manager = ConfigurationManager()

    # Create development configuration
    dev_config = {
        'data_dir': './data',
        'models_dir': './models',
        'logs_dir': './logs',
        'dashboard_port': 8501,
        'auto_backup': True,
        'quality_threshold': 0.8,
        'batch_size': 500,
        'log_level': 'DEBUG'
    }

    manager.create_configuration('development', 'cli', dev_config)

    # Get configuration
    config = manager.get_configuration('development')
    print("Development config:", json.dumps(config, indent=2))

    # Update configuration
    manager.update_configuration('development', {'batch_size': 1000})

    # Export configuration
    exported = manager.export_configuration('development', 'yaml')
    if exported:
        print("YAML export:")
        print(exported)