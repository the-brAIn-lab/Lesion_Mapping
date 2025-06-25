#!/bin/bash
# Model Backup and Versioning System
# Preserves your working baseline model before any experiments

echo "🔒 Model Backup and Versioning System"
echo "====================================="

# Set base directory
BASE_DIR="/mnt/beegfs/hellgate/home/rb194958e/stroke_segmentation_sota"
cd $BASE_DIR

# Create backup directory structure
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="model_backups/baseline_working_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

echo "📁 Creating backup directory: $BACKUP_DIR"

# Function to safely backup files
backup_file() {
    local source_file=$1
    local backup_name=$2
    
    if [ -f "$source_file" ]; then
        cp -p "$source_file" "$BACKUP_DIR/$backup_name"
        echo "✅ Backed up: $source_file → $BACKUP_DIR/$backup_name"
        
        # Calculate and store checksum
        md5sum "$source_file" > "$BACKUP_DIR/${backup_name}.md5"
    else
        echo "⚠️  File not found: $source_file"
    fi
}

# Backup current working models
echo ""
echo "🔄 Backing up working models..."

# Backup the main working models
backup_file "checkpoints/working_model.h5" "working_model.h5"
backup_file "checkpoints/final_working_model.h5" "final_working_model.h5"

# Backup any other checkpoint files
for checkpoint in checkpoints/*.h5; do
    if [ -f "$checkpoint" ]; then
        filename=$(basename "$checkpoint")
        backup_file "$checkpoint" "checkpoint_$filename"
    fi
done

# Backup training logs
echo ""
echo "📊 Backing up training logs..."
if [ -f "logs/training_1123608.out" ]; then
    backup_file "logs/training_1123608.out" "training_log.out"
fi

# Backup the working training script
echo ""
echo "📝 Backing up working scripts..."
backup_file "final_working_train.py" "final_working_train.py"
backup_file "scripts/final_working_training.sh" "final_working_training.sh"

# Create metadata file
echo ""
echo "📋 Creating backup metadata..."
cat > "$BACKUP_DIR/backup_metadata.txt" << EOF
Backup Information
==================
Date: $(date)
Job ID: 1123608
Model Type: Basic U-Net (Memory Efficient)
Parameters: ~500K-1M
Input Size: 128×128×128
Training Epochs: 50
Final Accuracy: 99.79%
Final Val Accuracy: 99.66%
Final Loss: 0.5590
Final Val Loss: 0.7560

Environment:
- Environment: tf215_env
- TensorFlow: 2.15.1
- GPU: RTX 4500 Ada Generation (24GB)
- CUDA: 12.6.3
- Data: ATLAS 2.0 (655 training samples)

Notes:
- This is the PROVEN WORKING baseline model
- No OOM errors with this configuration
- Successfully completed 50 epochs
- Good accuracy but possible overfitting (val_loss higher than train_loss)

Directory Structure:
$(ls -la "$BACKUP_DIR")

Checksums:
$(cat "$BACKUP_DIR"/*.md5 2>/dev/null)
EOF

# Create a symlink to latest backup
LATEST_LINK="model_backups/LATEST_WORKING_BASELINE"
rm -f "$LATEST_LINK"
ln -s "baseline_working_${TIMESTAMP}" "$LATEST_LINK"
echo "🔗 Created symlink: $LATEST_LINK → baseline_working_${TIMESTAMP}"

# Create recovery script
cat > "$BACKUP_DIR/restore_backup.sh" << 'RESTORE_SCRIPT'
#!/bin/bash
# Restore script for this backup

echo "🔄 Restoring model backup..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/../../.." && pwd )"

cd "$BASE_DIR"

# Verify checksums
echo "🔍 Verifying file integrity..."
for md5file in "$SCRIPT_DIR"/*.md5; do
    if [ -f "$md5file" ]; then
        md5sum -c "$md5file" || echo "⚠️  Checksum mismatch for $(basename $md5file .md5)"
    fi
done

# Restore files
echo ""
echo "📦 Restoring files..."
read -p "This will overwrite current models. Continue? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cp -v "$SCRIPT_DIR/working_model.h5" "checkpoints/working_model.h5"
    cp -v "$SCRIPT_DIR/final_working_model.h5" "checkpoints/final_working_model.h5"
    cp -v "$SCRIPT_DIR/final_working_train.py" "final_working_train.py"
    echo "✅ Restore completed!"
else
    echo "❌ Restore cancelled"
fi
RESTORE_SCRIPT

chmod +x "$BACKUP_DIR/restore_backup.sh"

# Create version control log
VERSION_LOG="model_backups/version_history.log"
cat >> "$VERSION_LOG" << EOF

================================================================================
Backup: baseline_working_${TIMESTAMP}
Date: $(date)
Type: Working Baseline
Accuracy: 99.79% / Val: 99.66%
Loss: 0.5590 / Val: 0.7560
Status: PROVEN WORKING - NO OOM
Notes: Original successful training run, 50 epochs completed
================================================================================
EOF

# Create a backup summary
echo ""
echo "📊 Backup Summary"
echo "================="
echo "Location: $BACKUP_DIR"
echo "Files backed up: $(ls -1 "$BACKUP_DIR" | grep -v "\.md5$" | grep -v "\.txt$" | grep -v "\.sh$" | wc -l)"
echo "Total size: $(du -sh "$BACKUP_DIR" | cut -f1)"
echo ""
echo "✅ Backup completed successfully!"
echo ""
echo "💡 To restore this backup later, run:"
echo "   $BACKUP_DIR/restore_backup.sh"
echo ""
echo "🔒 Your working model is now safely preserved!"
