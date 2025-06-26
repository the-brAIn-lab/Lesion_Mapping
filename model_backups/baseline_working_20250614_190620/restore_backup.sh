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
