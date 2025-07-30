#!/bin/bash
# Quick script to crop the original Atlas dataset to 128x128x128

echo "=== ATLAS DATASET CROPPING SCRIPT ==="
echo "This will crop the original full-resolution data to 128x128x128"
echo "Maintains image quality - only crops around center, no downsampling"
echo ""

# Set paths
IMAGES_DIR="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Images"
MASKS_DIR="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Masks"
OUTPUT_DIR="/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Cropped_128_Combined"

echo "Images directory: $IMAGES_DIR"
echo "Masks directory: $MASKS_DIR" 
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if directories exist
if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ Images directory not found: $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$MASKS_DIR" ]; then
    echo "❌ Masks directory not found: $MASKS_DIR"
    exit 1
fi

# Count files
IMG_COUNT=$(find "$IMAGES_DIR" -name "*_T1w.nii.gz" | wc -l)
MASK_COUNT=$(find "$MASKS_DIR" -name "*_mask.nii.gz" | wc -l)

echo "Found $IMG_COUNT image files"
echo "Found $MASK_COUNT mask files"
echo ""

if [ $IMG_COUNT -eq 0 ] || [ $MASK_COUNT -eq 0 ]; then
    echo "❌ No files found to process!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting cropping process..."
echo "This may take several minutes depending on dataset size."
echo ""

# Run the cropping script
python crop_original_dataset.py \
    --images_dir "$IMAGES_DIR" \
    --masks_dir "$MASKS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_size 128 \
    --max_workers 8

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "=== CROPPING COMPLETED ==="
    
    # Count output files
    OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*.nii.gz" | wc -l)
    IMG_OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*_T1w_*.nii.gz" | wc -l)
    MASK_OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*_mask_*.nii.gz" | wc -l)
    
    echo "Output files created: $OUTPUT_COUNT total"
    echo "  Images: $IMG_OUTPUT_COUNT"
    echo "  Masks: $MASK_OUTPUT_COUNT"
    echo ""
    
    echo "Sample output files:"
    ls -la "$OUTPUT_DIR" | head -10
    echo ""
    
    echo "✅ Dataset ready for training!"
    echo "Use this directory in your training script:"
    echo "DATA_DIR = Path('$OUTPUT_DIR')"
    
else
    echo "❌ Cropping failed. Check the logs for details."
    exit 1
fi
