# For bootstrapping test cases for transform/resample.
# Usage:
#   cd visor/tests
#   python bootstrapping_transform.py --bootstrap 0

# Or if you have 3dimg_cruncher cloned somewhere:
#   python bootstrapping_transform.py --bootstrap 1
# Ref. https://github.com/visor-tech/3dimg_cruncher/blob/main/img_resampling.py

import os
import numpy as np
import numpy.testing as npt  # for assert_allclose
import argparse

import SimpleITK as sitk

# Utility function to convert lists to numpy arrays
_a = lambda x: np.array(x, dtype=np.float64, order='C')

def Bootstrapping(test_slice_path):
    """Generate ideal affine transform and save it."""

    # Ref. SimpleITK Spatial Transformations:
    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/22_Transforms.html
    # For a stack of a slice
    #   From: Raw stack space (unit: voxel position):
    #     pos_raw = (frame_idx, y_raw, x_raw)
    #     e.g. (400, 787, 2047)
    #   To: Orthogonal stack space (unit: um, _st means stack):
    #     pos_stack = (z_stack, y_stack, x_stack)
    #       z_stack: advance direction,
    #       y_stack: thickness,
    #       x_stack: camera field width,
    #     e.g. (1414.21356237, 556.49303679, 2047)
    
    # set pixel size here, can be obtained from metadata in ome-zarr
    voxel_size_raw = _a((2.5, 1.0, 1.0))

    oblique_arcdeg = 45/180 * np.pi
    s = np.sin(oblique_arcdeg)
    c = np.cos(oblique_arcdeg)

    A_right = np.diag(voxel_size_raw) @ \
        _a([
            [1/s, 0, 0],
            [ -c, s, 0],
            [  0, 0, 1],
        ])

    # The setting of 788-1 is due to index starts from 0.
    # so that {frame_idx=0, y_raw=787} is {z_stack=0, y_stack=787/sqrt(2)}
    align_origin_to_y_lim = 788-1
    origin_shift = align_origin_to_y_lim * voxel_size_raw[1] * c
    offset_vec = _a([[origin_shift, 0, 0]])
    
    if 0:
        print("Affine transform matrix part (A_right):")
        print(A_right)
        print("Offset (offset_vec):")
        print(offset_vec)
    
    # Usage:
    # pos_stack = pos_raw @ A_right + offset_vec

    # affine transformation matrix (Transposed)
    # [[A_right   , 0],
    #  [offset_vec, 1]]

    transform_file_name = os.path.join(test_slice_path, "affine_transform.tfm")
    SaveAffineTransform(transform_file_name, A_right, offset_vec)

    T_raw_to_ortho = AffineTransform(A_right, offset_vec)

    return T_raw_to_ortho

def GenerateTestData(T_raw_to_ortho):
    """Generate test case for test_transform.py."""
    pos_raw_set = [
        _a([0, 0, 2047]),
        _a([
            [0, 0, 2047],
            [0, 787, 2047],
            [400, 787, 2047]])
    ]

    pos_stack_set = [
        T_raw_to_ortho.apply(pos_raw) for pos_raw in pos_raw_set
    ]
    for idx in range(len(pos_raw_set)):
        print(f"Test {idx}:")
        print(f"    pos_raw = {pos_raw_set[idx]}")
        print(f"    pos_stack = {pos_stack_set[idx]}")
    
    return pos_raw_set, pos_stack_set
    
def CompareWithOldCode(pos_raw_set, pos_stack_set):
    import sys
    sys.path.append('../../../3dimg_cruncher/flsmio')
    sys.path.append('../../../3dimg_cruncher')
    from img_resampling import frame_coor_to_fries_coor

    print("Comparing with old code...")
    for idx, pos_raw in enumerate(pos_raw_set):
        r_fries = frame_coor_to_fries_coor(pos_raw)
        print("*******************************************")
        print("Raw position:", pos_raw, sep='\n')
        print("Answer :", r_fries, sep='\n')
        npt.assert_allclose(pos_stack_set[idx], r_fries, rtol=1e-13, atol=2e-10)

class AffineTransform:
    """A class to represent a transform."""
    def __init__(self, A_right, offset_vec):
        self.A_right = A_right
        self.offset_vec = offset_vec

    def apply(self, pos_raw):
        """Apply the affine transformation to a raw position."""
        pos_stack = pos_raw @ self.A_right + self.offset_vec
        if (not hasattr(pos_raw, 'shape')) or len(pos_raw.shape) == 1:
            # keep the shape of pos_stack the same as pos_raw
            pos_stack = pos_stack[0]
        return pos_stack

def SaveAffineTransform(transform_file_name, A_right, offset_vec):
    """
    Save the affine transform matrix and offset vector in .tfm format.
    """
    affine_transform = sitk.AffineTransform(3)  # 3D affine transform
    
    # Set the matrix part (A_right)
    matrix_flat = A_right.T.flatten().tolist()
    affine_transform.SetMatrix(matrix_flat)
    
    translation = offset_vec[0].tolist()
    affine_transform.SetTranslation(translation)
    
    sitk.WriteTransform(affine_transform, transform_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test transform/resample functionality")
    parser.add_argument("--bootstrap", type=int, choices=[0, 1], 
                       help="Run bootstrapping test (0=False, 1=True)")
    args = parser.parse_args()

    test_slice_path = "./data/VISOR001.vsr/visor_recon_transforms/xxx_20250525/slice_1_10x"
    
    if args.bootstrap is not None:
        print("Bootstrapping test for transform/resample")
        T_raw_to_stack = Bootstrapping(test_slice_path)
        pos_raw_set, pos_stack_set = GenerateTestData(T_raw_to_stack)
        if args.bootstrap:
            # test against old code
            CompareWithOldCode(pos_raw_set, pos_stack_set)
    else:
        print("No action specified. Use --bootstrap 0 or --bootstrap 1")