import json
from pathlib import Path
import cv2
import numpy as np
from cens.cens_mask_patterns import canny_raw_magnitude


def add_mask_paths_to_transforms(transforms_json_path: str, images_dir: str = 'images', masks_dir: str = 'masks') -> int:
    """
    Add mask_path to each frame in a transforms.json file.
    
    Args:
        transforms_json_path: Path to the transforms.json file
        images_dir: Directory name to replace (default: 'images')
        masks_dir: Directory name to replace with (default: 'masks')
    
    Returns:
        Number of frames modified
    """
    # Read the JSON file
    with open(transforms_json_path, 'r') as f:
        data = json.load(f)
    
    # Add mask_path to each frame
    modified_count = 0
    for frame in data['frames']:
        if 'file_path' in frame:
            frame['mask_path'] = frame['file_path'].replace(images_dir, masks_dir)
            modified_count += 1
    
    # Write back to file
    with open(transforms_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✓ Modified {modified_count} frames in {Path(transforms_json_path).name}")
    print(f"Sample transformation:")
    print(f"  file_path: {data['frames'][0]['file_path']}")
    print(f"  mask_path: {data['frames'][0]['mask_path']}")
    
    return modified_count


def generate_sobel_masks(images_dir: str, masks_dir: str = None) -> int:
    """
    Generate high-frequency masks using Sobel edge detection from images.
    
    Args:
        images_dir: Path to the directory containing images
        masks_dir: Path to the directory to save masks (default: images_dir/../masks)
    
    Returns:
        Number of masks generated
    """
    images_path = Path(images_dir)
    
    # Default masks directory
    if masks_dir is None:
        masks_dir = images_path.parent / 'masks'
    else:
        masks_dir = Path(masks_dir)
    
    # Create masks directory if it doesn't exist
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files (common image formats)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [f for f in images_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"⚠ No images found in {images_path}")
        return 0
    
    generated_count = 0
    for img_path in sorted(image_files):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠ Failed to load {img_path.name}")
                continue
            
            # Generate sobel mask
            # the_mask = sobel_high_freq_mask(img)
            the_mask = canny_raw_magnitude(img) # use top 60% and top 80% gradients for thresholds
            
            # Save mask
            mask_output_path = masks_dir / img_path.name
            cv2.imwrite(str(mask_output_path), the_mask)
            generated_count += 1

        except Exception as e:
            print(f"⚠ Error processing {img_path.name}: {str(e)}")

    
    print(f"\n✓ Generated {generated_count} masks in {masks_dir}")
    return generated_count


def validate_and_setup_masks(mask_list, data_dir: str, transforms_json_path: str = None) -> dict:
    """
    Checks if mask_sobel (pattern index 3) is in use, and validates/creates masks and mask_path references.
    
    Args:
        mask_list: List of mask pattern indices from config
        data_dir: Base directory containing images folder (will look for transforms.json here)
        transforms_json_path: Optional explicit path to transforms.json (default: {data_dir}/transforms.json)
    
    Returns:
        Dictionary with status of checks and actions taken
    """
    status = {
        'mask_sobel_enabled': False,
        'masks_folder_exists': False,
        'mask_path_in_frames': False,
        'masks_generated': 0,
        'frames_updated': 0,
        'errors': []
    }
    
    data_path = Path(data_dir)
    images_path = data_path / 'images'
    masks_path = data_path / 'masks'
    
    # Determine transforms.json path
    if transforms_json_path is None:
        transforms_json_path = str(data_path / 'transforms.json')
    
    # Check if mask_sobel (index 3) is in mask_pattern
    if mask_list is not None:
        status['mask_sobel_enabled'] = 3 in mask_list
    
    if not status['mask_sobel_enabled']:
        print("ℹ Mask sobel (pattern 3) not enabled in config. Skipping mask setup.")
        return status
    
    print("\n" + "="*60)
    print("MASK SOBEL SETUP VALIDATION")
    print("="*60)
    
    # Check if transforms.json exists
    if not Path(transforms_json_path).exists():
        status['errors'].append(f"transforms.json not found at {transforms_json_path}")
        return status
    
    # Check if masks folder exists
    status['masks_folder_exists'] = masks_path.exists()
    
    # Check if mask_path exists in frames
    try:
        with open(transforms_json_path, 'r') as f:
            data = json.load(f)
        
        if data['frames']:
            status['mask_path_in_frames'] = 'mask_path' in data['frames'][0]
            
            # Count how many frames have mask_path
            frames_with_mask_path = sum(1 for frame in data['frames'] if 'mask_path' in frame)
            total_frames = len(data['frames'])
            
            if frames_with_mask_path != total_frames:
                status['mask_path_in_frames'] = False
    except Exception as e:
        status['errors'].append(f"Error reading transforms.json: {str(e)}")
        return status
    
    # Report findings
    print(f"\n✓ Mask Sobel enabled: {status['mask_sobel_enabled']}")
    print(f"{'✓' if status['masks_folder_exists'] else '✗'} Masks folder exists: {masks_path}")
    print(f"{'✓' if status['mask_path_in_frames'] else '✗'} mask_path in frames: {frames_with_mask_path}/{total_frames}" )
    
    # Generate masks if folder doesn't exist
    if not status['masks_folder_exists']:
        print(f"\n→ Creating masks folder and generating Sobel masks...")
        if not images_path.exists():
            status['errors'].append(f"Images folder not found at {images_path}")
            return status
        
        try:
            status['masks_generated'] = generate_sobel_masks(str(images_path), str(masks_path))
            status['masks_folder_exists'] = True
        except Exception as e:
            status['errors'].append(f"Error generating masks: {str(e)}")
            return status
    
    # Add mask_path to frames if missing
    if not status['mask_path_in_frames']:
        print(f"\n→ Adding mask_path references to frames...")
        try:
            status['frames_updated'] = add_mask_paths_to_transforms(transforms_json_path)
            status['mask_path_in_frames'] = True
        except Exception as e:
            status['errors'].append(f"Error updating transforms.json: {str(e)}")
            return status
    
    # Final summary
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    if status['errors']:
        print("❌ Errors encountered:")
        for error in status['errors']:
            print(f"  - {error}")
    else:
        print("✓ All validations passed!")
        if status['masks_generated'] > 0:
            print(f"✓ Generated {status['masks_generated']} masks")
        if status['frames_updated'] > 0:
            print(f"✓ Updated {status['frames_updated']} frame entries")
    print("="*60 + "\n")
    
    return status


if __name__ == '__main__':
    # Example usage - call validate_and_setup_masks with your config object
    # This should be called after your config is initialized in debug_script.py
    # 
    from debug_script import reduce_config
    from nerfstudio.configs.method_configs import all_methods

    # Note: reduce_config also saves a config.yml under <tmp_path>/outputs/<experiment>/cens/<timestamp>/.
    selected_config = reduce_config(all_methods["cens"], tmp_path=Path("/wkspace/"))

    # Check and setup masks if mask_sobel is enabled
    validate_and_setup_masks(selected_config.pipeline.datamanager.mask_pattern, "/wkspace/data/Frederik")
