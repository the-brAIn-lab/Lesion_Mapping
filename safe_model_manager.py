#!/usr/bin/env python3
"""
Safe Model Management System
Prevents accidental overwriting of working models
Implements versioning and backup strategies
"""

import os
import shutil
import hashlib
import json
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeModelManager:
    """Manages model versioning and prevents accidental overwrites"""
    
    def __init__(self, base_dir="/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota"):
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir / "model_backups"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.models_dir = self.base_dir / "models"
        
        # Create directories
        self.backup_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Protected model list - these should never be overwritten
        self.protected_models = [
            "checkpoints/working_model.h5",
            "checkpoints/final_working_model.h5"
        ]
        
        # Version tracking
        self.version_file = self.backup_dir / "model_versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self):
        """Load version history"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "backups": []}
    
    def _save_versions(self):
        """Save version history"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _calculate_checksum(self, filepath):
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_model_info(self, model_path):
        """Extract model information"""
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            info = {
                "parameters": model.count_params(),
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "layers": len(model.layers)
            }
            del model  # Free memory
            return info
        except:
            return {}
    
    def backup_model(self, model_path, description="", protect=False):
        """Create a backup of a model"""
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None
        
        # Create backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_path.stem
        backup_name = f"{model_name}_backup_{timestamp}.h5"
        backup_path = self.backup_dir / backup_name
        
        # Copy model
        shutil.copy2(model_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        # Calculate checksum
        checksum = self._calculate_checksum(backup_path)
        
        # Get model info
        model_info = self._get_model_info(backup_path)
        
        # Record backup
        backup_record = {
            "timestamp": timestamp,
            "original_path": str(model_path),
            "backup_path": str(backup_path),
            "checksum": checksum,
            "description": description,
            "protected": protect,
            "model_info": model_info
        }
        
        self.versions["backups"].append(backup_record)
        self._save_versions()
        
        # Mark as protected if requested
        if protect:
            self.protected_models.append(str(backup_path))
            logger.info(f"Model marked as protected: {backup_path}")
        
        return backup_path
    
    def save_model_safely(self, model, filepath, description="", force=False):
        """Save a model with automatic versioning"""
        filepath = Path(filepath)
        
        # Check if file exists and is protected
        if filepath.exists():
            full_path = str(self.base_dir / filepath)
            if any(full_path.endswith(protected) for protected in self.protected_models):
                if not force:
                    logger.error(f"Cannot overwrite protected model: {filepath}")
                    logger.info("Use force=True to override protection (not recommended)")
                    return None
                else:
                    logger.warning(f"FORCING overwrite of protected model: {filepath}")
                    # Create backup before overwriting
                    self.backup_model(filepath, f"Auto-backup before forced overwrite")
            else:
                # Create backup of existing model
                logger.info(f"Model exists, creating backup first...")
                self.backup_model(filepath, f"Auto-backup before overwrite")
        
        # Generate unique filename if needed
        if filepath.exists() and not force:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = filepath.stem
            suffix = filepath.suffix
            filepath = filepath.parent / f"{stem}_{timestamp}{suffix}"
            logger.info(f"Using versioned filename: {filepath}")
        
        # Save model
        try:
            model.save(filepath)
            logger.info(f"Model saved: {filepath}")
            
            # Calculate checksum
            checksum = self._calculate_checksum(filepath)
            
            # Get model info
            model_info = {
                "parameters": model.count_params(),
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "layers": len(model.layers)
            }
            
            # Update version tracking
            self.versions["models"][str(filepath)] = {
                "timestamp": datetime.now().isoformat(),
                "checksum": checksum,
                "description": description,
                "model_info": model_info
            }
            self._save_versions()
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
    
    def create_experiment_checkpoint(self, name, config=None):
        """Create a checkpoint for a new experiment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.checkpoints_dir / f"experiment_{name}_{timestamp}"
        experiment_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        if config:
            config_path = experiment_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # Create experiment info
        info_path = experiment_dir / "experiment_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Experiment: {name}\n")
            f.write(f"Created: {datetime.now()}\n")
            f.write(f"Base Directory: {experiment_dir}\n")
            f.write("\nProtected Models:\n")
            for model in self.protected_models:
                f.write(f"- {model}\n")
        
        logger.info(f"Created experiment checkpoint: {experiment_dir}")
        return experiment_dir
    
    def protect_current_models(self):
        """Protect all current working models"""
        protected_count = 0
        
        # Protect main working models
        for model_path in ["checkpoints/working_model.h5", 
                          "checkpoints/final_working_model.h5"]:
            full_path = self.base_dir / model_path
            if full_path.exists():
                backup_path = self.backup_model(
                    full_path, 
                    description="Protected working baseline",
                    protect=True
                )
                if backup_path:
                    protected_count += 1
        
        logger.info(f"Protected {protected_count} models")
        return protected_count
    
    def list_backups(self, show_details=False):
        """List all model backups"""
        print("\n📦 Model Backups")
        print("=" * 80)
        
        for backup in self.versions["backups"]:
            print(f"\n📁 {backup['backup_path']}")
            print(f"   Created: {backup['timestamp']}")
            print(f"   Original: {backup['original_path']}")
            print(f"   Protected: {'🔒 Yes' if backup['protected'] else '❌ No'}")
            
            if show_details and backup.get('model_info'):
                info = backup['model_info']
                print(f"   Parameters: {info.get('parameters', 'Unknown'):,}")
                print(f"   Input Shape: {info.get('input_shape', 'Unknown')}")
            
            if backup.get('description'):
                print(f"   Description: {backup['description']}")
    
    def restore_backup(self, backup_path, target_path=None):
        """Restore a model from backup"""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        # Find backup record
        backup_record = None
        for backup in self.versions["backups"]:
            if backup["backup_path"] == str(backup_path):
                backup_record = backup
                break
        
        if not backup_record:
            logger.warning("Backup not found in version history")
        
        # Determine target path
        if target_path is None and backup_record:
            target_path = backup_record["original_path"]
        elif target_path is None:
            logger.error("No target path specified")
            return False
        
        target_path = Path(target_path)
        
        # Check if target is protected
        if any(str(target_path).endswith(protected) for protected in self.protected_models):
            response = input(f"⚠️  Target is protected: {target_path}\nContinue? (y/N): ")
            if response.lower() != 'y':
                logger.info("Restore cancelled")
                return False
        
        # Create backup of current target if it exists
        if target_path.exists():
            self.backup_model(target_path, "Auto-backup before restore")
        
        # Restore
        shutil.copy2(backup_path, target_path)
        logger.info(f"✅ Restored: {backup_path} → {target_path}")
        
        return True


# Utility functions for easy usage
def protect_working_models():
    """Quick function to protect current working models"""
    manager = SafeModelManager()
    manager.protect_current_models()

def safe_save_model(model, filepath, description=""):
    """Save a model safely with automatic backup"""
    manager = SafeModelManager()
    return manager.save_model_safely(model, filepath, description)

def create_experiment(name, config=None):
    """Create a new experiment checkpoint"""
    manager = SafeModelManager()
    return manager.create_experiment_checkpoint(name, config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Model Management")
    parser.add_argument("--protect", action="store_true", help="Protect current models")
    parser.add_argument("--list", action="store_true", help="List all backups")
    parser.add_argument("--backup", type=str, help="Backup a specific model")
    parser.add_argument("--restore", type=str, help="Restore from backup")
    parser.add_argument("--target", type=str, help="Target path for restore")
    
    args = parser.parse_args()
    
    manager = SafeModelManager()
    
    if args.protect:
        manager.protect_current_models()
    elif args.list:
        manager.list_backups(show_details=True)
    elif args.backup:
        manager.backup_model(args.backup, description="Manual backup", protect=True)
    elif args.restore:
        manager.restore_backup(args.restore, args.target)
    else:
        # Default action: protect current models
        print("🔒 Protecting current working models...")
        manager.protect_current_models()
        print("\n💡 Use --list to see all backups")
        print("💡 Use --backup <model_path> to backup a specific model")
        print("💡 Use --restore <backup_path> to restore a backup")
