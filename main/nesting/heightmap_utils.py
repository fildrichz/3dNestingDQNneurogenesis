"""
Utilities for extracting heightmap patches for neural network input
"""
import numpy as np
from typing import List, Tuple

def extract_heightmap_patch(heightmap: np.ndarray, ex: int, ey: int, 
                            patch_size: int, resolution: int, 
                            container_W: int, container_D: int, container_H: int) -> np.ndarray:
    """
    Extract a patch_size x patch_size patch around position (ex, ey) from heightmap.
    
    Args:
        heightmap: (grid_W, grid_D) array of heights
        ex, ey: Position in container coordinates
        patch_size: Size of patch to extract (e.g., 7 for 7x7)
        resolution: Grid resolution (e.g., 10 means each grid cell is 10x10 container units)
        container_W, container_D, container_H: Container dimensions
    
    Returns:
        patch: (patch_size, patch_size) normalized heightmap patch
    """
    # Convert container coordinates to grid coordinates
    gx = ex // resolution
    gy = ey // resolution
    
    grid_W, grid_D = heightmap.shape
    half_patch = patch_size // 2
    
    # Initialize patch with zeros (ground level)
    patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    
    # Extract patch with boundary handling
    for i in range(patch_size):
        for j in range(patch_size):
            # Grid coordinates for this patch position
            ngx = gx - half_patch + i
            ngy = gy - half_patch + j
            
            # Boundary check
            if 0 <= ngx < grid_W and 0 <= ngy < grid_D:
                # Normalize height to [0, 1]
                patch[i, j] = heightmap[ngx, ngy] / container_H
            # else: leave as 0 (out of bounds = ground level)
    
    return patch


def extract_patches_for_actions(env, actions, patch_size: int = 7):
    """
    Extract heightmap patches for all actions in the current action space.
    
    Args:
        env: MultiBinPackingEnv instance
        actions: List of action tuples from env.enumerate_actions()
        patch_size: Size of patches to extract
    
    Returns:
        patches: (num_actions, patch_size, patch_size) array of patches
    """
    W, D, H = env.bin_size
    patches = []
    
    for action in actions:
        if action is None:
            # Padding action - return zero patch
            patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
        else:
            bin_idx, item_idx, ep_idx, rot_idx, ep, size, weight, item_id = action
            target_bin = env.bins[bin_idx]
            ex, ey, ez = ep
            
            # Extract patch from this bin's heightmap
            patch = extract_heightmap_patch(
                target_bin.heightmap, 
                ex, ey, 
                patch_size, 
                target_bin.resolution,
                W, D, H
            )
            patches.append(patch)
    
    return np.array(patches, dtype=np.float32)


def pad_patches(patches: np.ndarray, max_actions: int, patch_size: int) -> np.ndarray:
    """
    Pad patches array to max_actions size.
    
    Args:
        patches: (A, patch_size, patch_size) array
        max_actions: Target size
        patch_size: Patch dimensions
    
    Returns:
        padded: (max_actions, patch_size, patch_size) array
    """
    A = patches.shape[0]
    padded = np.zeros((max_actions, patch_size, patch_size), dtype=np.float32)
    padded[:A] = patches[:A]
    return padded
