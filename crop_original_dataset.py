#!/usr/bin/env python3
"""
Crop Original Atlas Dataset to 128x128x128 around Center
Combines Images and Masks into single directory for training
Maintains full image quality - only crops, doesn't downsample
"""

import os
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def crop_center_3d(volume, target_shape=(128, 128, 128)):
    """
    Crop volume to target shape around the center
    Maintains image quality - no resampling, just cropping
    """
    current_shape = volume.shape
    
    # Calculate start and end indices for each dimension
    start_indices = []
    end_indices = []
    
    for i in range(3):
        current_size = current_shape[i]
        target_size = target_shape[i]
        
        if current_size >= target_size:
            # Center crop
            start = (current_size - target_size) // 2
            end = start + target_size
        else:
            # Pad if current size is smaller than target
            start = 0
            end = current_size
            
        start_indices.append(start)
        end_indices.append(end)
    
    # Crop the volume
    cropped = volume[
        start_indices[0]:end_indices[0],
        start_indices[1]:end_indices[1],
        start_indices[2]:end_indices[2]
    ]
    
    # Pad if necessary to reach exact target shape
    if cropped.shape != target_shape:
        padded = np.zeros(target_shape, dtype=cropped.dtype)
        
        # Calculate padding offsets
        pad_start = []
        for i in range(3):
            if cropped.shape[i] < target_shape[i]:
                pad_offset = (target_shape[i] - cropped.shape[i]) // 2
            else:
                pad_offset = 0
            pad_start.append(pad_offset)
        
        # Place cropped volume in center of padded array
        padded[
            pad_start[0]:pad_start[0]+cropped.shape[0],
            pad_start[1]:pad_start[1]+cropped.shape[1],
            pad_start[2]:pad_start[2]+cropped.shape[2]
        ] = cropped
        
        cropped = padded
    
    return cropped

def process_image_mask_pair(image_path, mask_path, output_dir, target_shape=(128, 128, 128)):
    """
    Process a single image-mask pair: crop and save to output directory
    """
    try:
        # Extract base filename for matching
        base_name = image_path.stem.replace('_T1w', '')
        
        # Load image
        img_nii = nib.load(str(image_path))
        img_data = img_nii.get_fdata()
        
        # Load mask
        mask_nii = nib.load(str(mask_path))
        mask_data = mask_nii.get_fdata()
        
        logger.info(f"Processing {base_name}")
        logger.info(f"  Original image shape: {img_data.shape}")
        logger.info(f"  Original mask shape: {mask_data.shape}")
        
        # Verify shapes match
        if img_data.shape != mask_data.shape:
            logger.warning(f"  Shape mismatch: img {img_data.shape} vs mask {mask_data.shape}")
            # Use the smaller shape as reference
            min_shape = tuple(min(img_data.shape[i], mask_data.shape[i]) for i in range(3))
            img_data = img_data[:min_shape[0], :min_shape[1], :min_shape[2]]
            mask_data = mask_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Crop both to target shape
        img_cropped = crop_center_3d(img_data, target_shape)
        mask_cropped = crop_center_3d(mask_data, target_shape)
        
        logger.info(f"  Cropped shape: {img_cropped.shape}")
        
        # Create output filenames
        img_output_name = f"{base_name}_T1w_cropped128.nii.gz"
        mask_output_name = f"{base_name}_mask_cropped128.nii.gz"
        
        # Save cropped image
        img_output_path = output_dir / img_output_name
        img_nii_cropped = nib.Nifti1Image(img_cropped, img_nii.affine, img_nii.header)
        nib.save(img_nii_cropped, str(img_output_path))
        
        # Save cropped mask
        mask_output_path = output_dir / mask_output_name
        mask_nii_cropped = nib.Nifti1Image(mask_cropped, mask_nii.affine, mask_nii.header)
        nib.save(mask_nii_cropped, str(mask_output_path))
        
        # Verify lesion presence
        lesion_voxels = np.sum(mask_cropped > 0)
        logger.info(f"  Lesion voxels in cropped mask: {lesion_voxels}")
        
        return {
            'base_name': base_name,
            'success': True,
            'original_shape': img_data.shape,
            'cropped_shape': img_cropped.shape,
            'lesion_voxels': lesion_voxels,
            'img_file': img_output_name,
            'mask_file': mask_output_name
        }
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return {
            'base_name': image_path.stem,
            'success': False,
            'error': str(e)
        }

def find_matching_pairs(images_dir, masks_dir):
    """
    Find matching image-mask pairs based on filenames
    Updated to handle Atlas naming convention
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # Get all image files - look for T1w images
    image_files = list(images_dir.glob("*_T1w.nii.gz"))
    # Get all mask files - look for lesion masks
    mask_files = list(masks_dir.glob("*_label-L_desc-T1lesion_mask.nii.gz"))
    
    logger.info(f"Found {len(image_files)} image files")
    logger.info(f"Found {len(mask_files)} mask files")
    
    # Show sample filenames for debugging
    if image_files:
        logger.info(f"Sample image: {image_files[0].name}")
    if mask_files:
        logger.info(f"Sample mask: {mask_files[0].name}")
    
    pairs = []
    unmatched_images = []
    unmatched_masks = []
    
    for img_file in image_files:
        # Extract base identifier by removing the T1w suffix
        # Example: sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
        # Becomes: sub-r001s001_ses-1_space-MNI152NLin2009aSym
        img_base = img_file.stem.replace('.nii', '').replace('_T1w', '')
        
        # Look for matching mask with the pattern:
        # sub-r001s001_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz
        expected_mask_name = f"{img_base}_label-L_desc-T1lesion_mask.nii.gz"
        expected_mask_path = masks_dir / expected_mask_name
        
        if expected_mask_path.exists():
            pairs.append((img_file, expected_mask_path))
        else:
            # Fallback: try to find any mask containing the base name
            matching_masks = [m for m in mask_files if img_base in m.stem]
            if matching_masks:
                pairs.append((img_file, matching_masks[0]))
            else:
                unmatched_images.append(img_file)
    
    # Find unmatched masks
    matched_mask_files = {pair[1] for pair in pairs}
    unmatched_masks = [m for m in mask_files if m not in matched_mask_files]
    
    logger.info(f"Successfully matched {len(pairs)} image-mask pairs")
    
    if unmatched_images:
        logger.warning(f"Unmatched images: {len(unmatched_images)}")
        for img in unmatched_images[:5]:  # Show first 5
            logger.warning(f"  {img.name}")
    
    if unmatched_masks:
        logger.warning(f"Unmatched masks: {len(unmatched_masks)}")
        for mask in unmatched_masks[:5]:  # Show first 5
            logger.warning(f"  {mask.name}")
    
    # Show first few successful pairs for verification
    if pairs:
        logger.info("Sample matched pairs:")
        for i, (img, mask) in enumerate(pairs[:3]):
            logger.info(f"  {i+1}: {img.name}")
            logger.info(f"     -> {mask.name}")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description='Crop Atlas dataset to 128x128x128')
    parser.add_argument('--images_dir', 
                       default='/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Images',
                       help='Directory containing original images')
    parser.add_argument('--masks_dir', 
                       default='/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Masks',
                       help='Directory containing original masks')
    parser.add_argument('--output_dir',
                       default='/mnt/beegfs/hellgate/home/rb194958e/Atlas_2/Training/Cropped_128_Combined',
                       help='Output directory for cropped data')
    parser.add_argument('--target_size', type=int, default=128,
                       help='Target crop size (default: 128 for 128x128x128)')
    parser.add_argument('--max_workers', type=int, default=8,
                       help='Maximum number of parallel workers')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only show what would be processed, don\'t actually crop')
    
    args = parser.parse_args()
    
    # Setup directories
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)
    
    if not masks_dir.exists():
        logger.error(f"Masks directory not found: {masks_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_shape = (args.target_size, args.target_size, args.target_size)
    
    logger.info(f"=== ATLAS DATASET CROPPING ===")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Masks directory: {masks_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target shape: {target_shape}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Find matching pairs
    pairs = find_matching_pairs(images_dir, masks_dir)
    
    if not pairs:
        logger.error("No matching image-mask pairs found!")
        sys.exit(1)
    
    if args.dry_run:
        logger.info("DRY RUN - Would process the following pairs:")
        for i, (img, mask) in enumerate(pairs[:10]):  # Show first 10
            logger.info(f"  {i+1}: {img.name} + {mask.name}")
        if len(pairs) > 10:
            logger.info(f"  ... and {len(pairs) - 10} more pairs")
        return
    
    # Process pairs
    logger.info(f"Processing {len(pairs)} image-mask pairs...")
    
    results = []
    successful = 0
    failed = 0
    
    if args.max_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(process_image_mask_pair, img, mask, output_dir, target_shape): (img, mask)
                for img, mask in pairs
            }
            
            # Process completed tasks
            with tqdm(total=len(pairs), desc="Processing") as pbar:
                for future in as_completed(future_to_pair):
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        successful += 1
                    else:
                        failed += 1
                    
                    pbar.update(1)
    else:
        # Sequential processing
        for img, mask in tqdm(pairs, desc="Processing"):
            result = process_image_mask_pair(img, mask, output_dir, target_shape)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
    
    # Summary
    logger.info(f"=== PROCESSING COMPLETE ===")
    logger.info(f"Successfully processed: {successful}/{len(pairs)}")
    logger.info(f"Failed: {failed}/{len(pairs)}")
    
    # Statistics
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        total_lesion_voxels = sum(r['lesion_voxels'] for r in successful_results)
        lesion_cases = sum(1 for r in successful_results if r['lesion_voxels'] > 0)
        
        logger.info(f"Total lesion voxels across all cases: {total_lesion_voxels}")
        logger.info(f"Cases with lesions: {lesion_cases}/{successful}")
        logger.info(f"Cases without lesions: {successful - lesion_cases}/{successful}")
        
        # Check output directory
        output_files = list(output_dir.glob("*.nii.gz"))
        logger.info(f"Files created in output directory: {len(output_files)}")
        logger.info(f"Expected files: {successful * 2} (image + mask per case)")
        
        # Show first few files
        logger.info("Sample output files:")
        for f in sorted(output_files)[:10]:
            logger.info(f"  {f.name}")
    
    if failed > 0:
        logger.warning("Failed cases:")
        failed_results = [r for r in results if not r['success']]
        for r in failed_results[:5]:  # Show first 5 failures
            logger.warning(f"  {r['base_name']}: {r.get('error', 'Unknown error')}")
    
    logger.info(f"Processing complete! Output directory: {output_dir}")

if __name__ == "__main__":
    main()
