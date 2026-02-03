import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import map_coordinates
from pathlib import Path

def random_rotation_3d(volume):
    """Apply random 3D rotation with optimized parameters."""
    angle_x = np.random.uniform(0, 360)
    angle_y = np.random.uniform(0, 360)
    angle_z = np.random.uniform(0, 360)
    
    volume = ndimage.rotate(volume, angle_x, axes=(1, 2), reshape=False, order=0)
    volume = ndimage.rotate(volume, angle_y, axes=(0, 2), reshape=False, order=0)
    volume = ndimage.rotate(volume, angle_z, axes=(0, 1), reshape=False, order=0)
    
    return volume

def random_flip_3d(volume):
    """Apply random flips along each axis."""
    if np.random.random() > 0.5:
        volume = np.flip(volume, axis=0)
    if np.random.random() > 0.5:
        volume = np.flip(volume, axis=1)
    if np.random.random() > 0.5:
        volume = np.flip(volume, axis=2)
    return volume.copy()

def elastic_deformation_3d(volume, alpha=10, sigma=3):
    """Apply elastic deformation to 3D volume with optimizations."""
    shape = volume.shape
    
    dx = ndimage.gaussian_filter(
        (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
    ) * alpha
    dy = ndimage.gaussian_filter(
        (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
    ) * alpha
    dz = ndimage.gaussian_filter(
        (np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
    ) * alpha
    
    z, y, x = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )
    
    indices = [
        np.clip(z + dz, 0, shape[0] - 1),
        np.clip(y + dy, 0, shape[1] - 1),
        np.clip(x + dx, 0, shape[2] - 1)
    ]
    
    deformed = map_coordinates(volume, indices, order=0, mode='nearest')
    return deformed

def random_scale_3d(volume, scale_range=(0.8, 1.2)):
    """Apply random isotropic scaling with optimized interpolation."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    scaled = ndimage.zoom(volume, scale_factor, order=0)
    return scaled

def crop_to_bbox(volume, padding=5):
    """Crop volume to bounding box of foreground with padding."""
    coords = np.where(volume > 0)
    if len(coords[0]) == 0:
        return volume
    
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    
    z_min = max(0, z_min - padding)
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    z_max = min(volume.shape[0] - 1, z_max + padding)
    y_max = min(volume.shape[1] - 1, y_max + padding)
    x_max = min(volume.shape[2] - 1, x_max + padding)
    
    cropped = volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    return cropped

def resize_to_fit(volume, max_size):
    """Resize volume if any dimension exceeds max_size."""
    shape = volume.shape
    if all(s <= max_size for s in shape):
        return volume
    
    max_dim = max(shape)
    scale_factor = max_size / max_dim
    
    scaled = ndimage.zoom(volume, scale_factor, order=0)
    scaled = (scaled > 0.5).astype(np.uint8) * 255
    
    return scaled

def augment_and_place_single_cell(single_cell_volume,
                                   output_shape=(1024, 256, 256),
                                   elastic_alpha=10,
                                   elastic_sigma=3,
                                   scale_range=(0.8, 1.2),
                                   max_cell_size=200):
    """Augment a single cell and place it randomly in specified volume shape.
    
    Returns:
        output_volume: Binary mask of the augmented cell
        position: (z_pos, y_pos, x_pos) tuple for tracking
    """
    
    augmented = single_cell_volume.copy()
    
    augmented = crop_to_bbox(augmented, padding=5)
    augmented = random_flip_3d(augmented)
    augmented = random_rotation_3d(augmented)
    augmented = random_scale_3d(augmented, scale_range)
    augmented = elastic_deformation_3d(augmented, elastic_alpha, elastic_sigma)
    
    augmented = (augmented > 0.5).astype(np.uint8) * 255
    augmented = crop_to_bbox(augmented, padding=2)
    augmented = resize_to_fit(augmented, max_cell_size)
    
    cz, cy, cx = augmented.shape
    out_z, out_y, out_x = output_shape
    
    if cz > out_z or cy > out_y or cx > out_x:
        max_allowed = min(out_z - 10, out_y - 10, out_x - 10)
        augmented = resize_to_fit(augmented, max_allowed)
        cz, cy, cx = augmented.shape
    
    output_volume = np.zeros(output_shape, dtype=np.uint8)
    
    z_pos = np.random.randint(0, max(1, out_z - cz + 1))
    y_pos = np.random.randint(0, max(1, out_y - cy + 1))
    x_pos = np.random.randint(0, max(1, out_x - cx + 1))
    
    output_volume[z_pos:z_pos+cz, y_pos:y_pos+cy, x_pos:x_pos+cx] = augmented
    
    return output_volume, (z_pos, y_pos, x_pos, cz, cy, cx)

def generate_multicell_volume(single_cell_volume,
                              n_cells=10,
                              output_shape=(1024, 256, 256),
                              elastic_alpha=10,
                              elastic_sigma=3,
                              scale_range=(0.8, 1.2),
                              max_cell_size=200):
    """Generate multi-cell volume with both synthetic image and label ground truth.
    
    Returns:
        synthetic_volume: The combined image (overlaps use maximum intensity)
        label_volume: Ground truth labels (1, 2, 3, ... for each cell)
    """
    
    synthetic_volume = np.zeros(output_shape, dtype=np.uint8)
    label_volume = np.zeros(output_shape, dtype=np.uint16) 
    
    for i in range(n_cells):
        cell_volume, position = augment_and_place_single_cell(
            single_cell_volume,
            output_shape=output_shape,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            scale_range=scale_range,
            max_cell_size=max_cell_size
        )
        
        # For synthetic image: use maximum (simulate overlapping intensities)
        np.maximum(synthetic_volume, cell_volume, out=synthetic_volume)
        
        # For label image: assign unique label where cell exists
        # Policy: last cell wins in overlapping regions
        cell_mask = cell_volume > 0
        label_volume[cell_mask] = i + 1  # Labels start at 1 (0 = background)
    
    return synthetic_volume, label_volume

def generate_multicell_dataset(input_path,
                                output_dir,
                                n_volumes=100,
                                cells_per_volume=10,
                                output_shape=(1024, 256, 256),
                                elastic_alpha=10,
                                elastic_sigma=3,
                                scale_range=(0.8, 1.2),
                                max_cell_size=200,
                                random_seed=None):
    """Generate dataset of synthetic multi-cell volumes with ground truth labels.
    
    Creates two subdirectories:
        - images/: synthetic microscopy images
        - labels/: corresponding ground truth segmentation labels
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load single cell
    print(f"Loading single cell mesh from {input_path}...")
    single_cell = tifffile.imread(input_path)
    single_cell = (single_cell > 0).astype(np.uint8) * 255
    print(f"Original cell shape: {single_cell.shape}")
    print(f"Original cell voxels: {np.sum(single_cell > 0)}")
    print(f"Output shape: {output_shape} (Z×Y×X)")
    
    base_name = Path(input_path).stem
    
    # Generate volumes sequentially
    print("\n" + "="*60)
    for i in range(n_volumes):
        print(f"\nGenerating volume {i+1}/{n_volumes}...")
        
        if random_seed is not None:
            np.random.seed(random_seed + i)
        
        synthetic_volume, label_volume = generate_multicell_volume(
            single_cell,
            n_cells=cells_per_volume,
            output_shape=output_shape,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            scale_range=scale_range,
            max_cell_size=max_cell_size
        )
        
        foreground_voxels = np.sum(synthetic_volume > 0)
        occupancy = foreground_voxels / synthetic_volume.size * 100
        n_labeled_cells = len(np.unique(label_volume)) - 1  # Subtract background
        
        # Save both images and labels
        image_path = images_dir / f"{base_name}_multicell_{i:04d}.tif"
        label_path = labels_dir / f"{base_name}_multicell_{i:04d}_labels.tif"
        
        tifffile.imwrite(image_path, synthetic_volume)
        tifffile.imwrite(label_path, label_volume)
        
        print(f"  Cells placed: {n_labeled_cells}")
        print(f"  Total foreground voxels: {foreground_voxels}")
        print(f"  Volume occupancy: {occupancy:.2f}%")
        print(f"  Saved image: {image_path}")
        print(f"  Saved labels: {label_path}")
    
    print(f"\n✓ Generated {n_volumes} synthetic multi-cell volumes with labels!")
    print(f"   Shape: {output_shape} (Z×Y×X)")
    print(f"   Images saved to: {images_dir}")
    print(f"   Labels saved to: {labels_dir}")


if __name__ == "__main__":
    generate_multicell_dataset(
        input_path="filled_mesh_volume.tif",
        output_dir="synthetic_microglia_dataset_v24",
        n_volumes=5,
        cells_per_volume=10,
        output_shape=(1024, 256, 256),
        elastic_alpha=10,
        elastic_sigma=3,
        scale_range=(0.3, 0.7),
        max_cell_size=150,
        random_seed=42
    )
