import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from scipy import fftpack
from svd import custom_svd
import time


def load_image(image_path):
    """Simple image loading function."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image.astype(np.float32) / 255.0

def fast_svd_filter(image, k=50):
    """Fast SVD-based image filtering."""
    U, sigma, Vt = custom_svd(image)

    
    
    sigma_k = np.zeros_like(sigma)
    sigma_k[:k] = sigma[:k]
    
    return (U @ np.diag(sigma_k) @ Vt).astype(np.float32)

def compute_normal_map(image):
    """Compute normal map directly from image using sobel filters."""
    
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    height, width = image.shape
    normal_map = np.zeros((height, width, 3), dtype=np.float32)
    normal_map[:, :, 0] = -dx
    normal_map[:, :, 1] = -dy
    normal_map[:, :, 2] = 1.0
    
    magnitudes = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    magnitudes = np.maximum(magnitudes, 1e-10) 
    normal_map = normal_map / magnitudes
    
    return normal_map

def calculate_shading(normal_map, light_direction=np.array([0.3, 0.1, 1.0])):
    """Calculate shading using dot product."""
    
    light_direction = light_direction / np.linalg.norm(light_direction)
    light_direction = light_direction.reshape(1, 1, 3)
    
    dot_product = np.sum(normal_map * light_direction, axis=2)
    return np.maximum(0, dot_product)

def poisson_solve(p, q):
    """
    Fast Poisson solver using FFT for height field integration.
    
    Parameters:
    p, q - gradient fields
    
    Returns:
    h - integrated height field
    """
    h, w = p.shape
    
    div = np.zeros_like(p)
    div[1:, :] -= p[:-1, :]  
    div[:, 1:] -= q[:, :-1]  
    div[:-1, :] += p[1:, :] 
    div[:, :-1] += q[:, 1:] 
    
    div_fft = fftpack.fft2(div)
    
    fy = fftpack.fftfreq(h)[:, np.newaxis]
    fx = fftpack.fftfreq(w)
    
    denom = 2 * np.cos(2 * np.pi * fx) + 2 * np.cos(2 * np.pi * fy) - 4
    denom[0, 0] = 1  
    height_fft = div_fft / denom
    height_fft[0, 0] = 0  
    height = np.real(fftpack.ifft2(height_fft))
    
    height = height - np.min(height)
    max_height = np.max(height)
    if max_height > 0:
        height = height / max_height
    
    return height

def fast_shape_from_shading(image, light_direction=np.array([0.3, 0.1, 1.0])):
    """Fast shape from shading algorithm."""
    
    normal_map = compute_normal_map(image)
    
    shading = calculate_shading(normal_map, light_direction)
    
    nx = normal_map[:, :, 0]
    ny = normal_map[:, :, 1]
    nz = normal_map[:, :, 2]
    
    epsilon = 1e-5
    p = -nx / (nz + epsilon)  
    q = -ny / (nz + epsilon)  
    
    depth_map = poisson_solve(p, q)
    
    return normal_map, shading, depth_map

def plot_terrain_3d(depth_map, subsample=4, scale_factor=0.5, cut_level=None):
    """Plot 3D terrain with optional height cutoff."""
    height, width = depth_map.shape
    
    y_indices = np.arange(0, height, subsample)
    x_indices = np.arange(0, width, subsample)
    subsampled_depth = depth_map[::subsample, ::subsample]
    
    subsampled_depth = subsampled_depth * scale_factor
    
    if cut_level is not None:
        subsampled_depth = np.ma.masked_where(subsampled_depth < cut_level, subsampled_depth)
    
    X, Y = np.meshgrid(x_indices, y_indices)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    surf = ax.plot_surface(
        X, Y, subsampled_depth,
        cmap='terrain',
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1
    )
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    
    ax.view_init(elev=35, azim=45)
    
    plt.tight_layout()
    plt.savefig("result_data/terrain_3d.png", dpi=300)
    plt.show()

def main(image_path, k=50, cut_level=None):
    """Main function to process an image and generate terrain."""
    print(f"Loading and preprocessing image: {image_path}")
    image = load_image(image_path)
    filtered_image = fast_svd_filter(image, k)
    
    print("Generating terrain from image...")
    normal_map, shading, depth_map = fast_shape_from_shading(filtered_image)
    
    np.save("result_data/depth_map.npy", depth_map)
    
    print("Creating 3D visualization...")
    plot_terrain_3d(depth_map, cut_level=cut_level)
    
    print("Done! The terrain has been reconstructed and saved.")
    return depth_map


if __name__ == "__main__":

    # image_path = "src_images/0.png"
    # main(image_path, k=50, cut_level=0.2) 

    def measure_performance(image_path, k=50, cut_level=None):
        start_time = time.time()
        depth_map = main(image_path, k, cut_level)
        end_time = time.time()
        
        execution_time = end_time - start_time
        surface_std = np.std(depth_map)
        
        return execution_time, surface_std

    
    image_paths = ["src_images/image0.png", "src_images/image1.png"]
    names = ["Image 0", "Image 1"]

    times = []
    stds = []


    for path in image_paths:
        image = load_image(path)
        
        exec_time, std_dev = measure_performance(path, k=50, cut_level=0.2)
        
        times.append(exec_time)
        stds.append(std_dev)

    def plot_metrics(names, times, stds, edge_densities, entropies):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Execution time
        axes[0].bar(names, times, color='skyblue')
        axes[0].set_title("Execution Time")
        axes[0].set_ylabel("Time (s)")

        # Surface variation
        axes[1].bar(names, stds, color='lightgreen')
        axes[1].set_title("Surface Variation (std dev of depth)")
        axes[1].set_ylabel("Depth Std Dev")

        plt.tight_layout()
        plt.show()

