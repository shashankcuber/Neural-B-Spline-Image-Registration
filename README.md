# Neural-Bspline Registration Algorithm

A deep learning-based medical image registration approach using B-spline transformation.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch (version requirements TBD)
- Other dependencies (TBD)

### Installation

Clone the repository:
```bash
git clone https://github.com/your-username/neural-bspline.git
cd neural-bspline
```

### Running the Algorithm

1. Train the model using:
```bash
python3 train.py
```

2. Configuring Control Points:
   - Open `train.py`
   - Locate the `train_config` dictionary
   - Modify the `num_points` parameter to change the number of control points

## Dataset Structure

The algorithm works with two main datasets:

### FIRE Dataset
```
Fire/
├── moving/
│   └── [moving image files]
└── fixed/
    └── [fixed image files]
```
Contains pairs of moving and fixed images from the FIRE FUNDUS Dataset.

### FUNDUS Segmentation Dataset
```
FUNDUS_seg_output/
├── moving/
│   └── [segmentation masks]
└── fixed/
    └── [segmentation masks]
```
Contains blood vessel segmentation masks for both fixed and moving images (FIRE-segmented dataset).

## Output

The algorithm saves all plots, figures, and their corresponding control points in:
```
plots/test-1/<num_points>/
```
where `<num_points>` corresponds to the number of control points used in the registration.

## Baseline Comparison

This implementation includes a comparison with VoxelMorph:

- Base implementation: [VoxelMorph-PyTorch](https://github.com/Hsankesara/VoxelMorph-PyTorch)
- Modification: Replaced NCC loss with MSE loss for comparison purposes


## Acknowledgments

- VoxelMorph implementation by [Hsankesara](https://github.com/Hsankesara/VoxelMorph-PyTorch)
