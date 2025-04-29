# linear_project

To run script

```
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python3 reconstruction.py
```

# Shape-from-Shading Terrain Reconstruction

This project implements a fast shape-from-shading algorithm to reconstruct 3D terrain from 2D grayscale images. It uses SVD-based image filtering, gradient-based normal map estimation, shading simulation, and a Poisson solver for surface integration.

## Features
- Custom SVD image compression and filtering
- Normal map computation from image gradients
- Lambertian shading simulation
- Fast Poisson solver using FFT for depth map recovery
- 3D terrain visualization with matplotlib

## Requirements
- Python 3.7+
- NumPy
- OpenCV (cv2)
- Matplotlib
- SciPy

## Usage
1. Place your image(s) in the `src_images/` directory.
2. Run the script:
```bash
python your_script_name.py
```
3. Results will be saved to the `result_data/` directory, including 3D plots and depth map data.

## Example
```python
image_path = "src_images/image0.png"
main(image_path, k=50, cut_level=0.2)
```

## Performance Evaluation
The script also includes performance metrics:
- Execution time
- Surface standard deviation (variation in depth)

Metrics are plotted for visual comparison.

## File Overview
- `custom_svd`: Custom singular value decomposition logic (imported from `svd.py`)
- `main()`: Entry point for processing and visualization
- `plot_terrain_3d()`: Renders 3D terrain
- `fast_shape_from_shading()`: Core shading-to-shape algorithm

## Output
- `terrain_3d.png`: 3D plot of the reconstructed surface
- `depth_map.npy`: Raw depth data for further analysis
---

