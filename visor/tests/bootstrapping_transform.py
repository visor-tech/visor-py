# For bootstrapping test cases for transform/resample.
# Usage:
#   cd visor/tests
#   python bootstrapping_transform.py --bootstrap 0

# Or if you have 3dimg_cruncher cloned somewhere:
#   python bootstrapping_transform.py --bootstrap 1
# Ref. https://github.com/visor-tech/3dimg_cruncher/blob/main/img_resampling.py

import os
import sys
import numpy as np
import numpy.testing as npt  # for assert_allclose
import argparse
import json
import pprint

import zarr
import SimpleITK as sitk

# Utility function to convert lists to numpy arrays
_a = lambda x: np.array(x, dtype=np.float64, order='C')

def BootstrappingTransform(test_slice_path):
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

    transform_file_name = os.path.join(test_slice_path, "raw_to_ortho", "affine_transform.tfm")
    os.makedirs(os.path.join(test_slice_path, "raw_to_ortho"), exist_ok=True)
    SaveAffineTransform(transform_file_name, A_right, offset_vec)
    CorrectTransformJson(test_slice_path)

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

    def _print_named_array(name, arr):
        str = np.array2string(arr, prefix = name + ' = ', precision=16, separator=', ')
        print(f"{name} = {str}")

    print("")
    print("Generated test data")
    print("===================")
    for idx in range(len(pos_raw_set)):
        print(f"Test {idx}:")
        _print_named_array("pos_raw", pos_raw_set[idx])
        _print_named_array("pos_stack", pos_stack_set[idx])
    
    return pos_raw_set, pos_stack_set
    
def CompareWithOldCode(pos_raw_set, pos_stack_set):
    sys.path.append('../../../3dimg_cruncher/flsmio')
    sys.path.append('../../../3dimg_cruncher')
    from img_resampling import frame_coor_to_fries_coor

    print("")
    print("Comparing with old code")
    print("=======================")
    print("(No assertion is good)")
    for idx, pos_raw in enumerate(pos_raw_set):
        r_fries = frame_coor_to_fries_coor(pos_raw)
        print("")
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

    def inverse(self):
        """Return the inverse of the affine transformation."""
        A_inv = np.linalg.inv(self.A_right)
        offset_vec_inv = -self.offset_vec @ A_inv
        return AffineTransform(A_inv, offset_vec_inv)

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
    
    print("transform_file_name:", transform_file_name)
    sitk.WriteTransform(affine_transform, transform_file_name)
    print("Saved affine transform to:", transform_file_name)

def CorrectTransformJson(test_slice_path, overwrite = False):
    # also correct the transforms.json
    T_set_metadata = json.load(open(os.path.join(test_slice_path, "transforms.json"), 'r'))
    #pprint.pprint(T_set_metadata)
    
    # Find and update the raw_to_ortho transform format
    found = False
    needs_update = False
    for transform in T_set_metadata:
        if transform.get('name') == 'raw_to_ortho':
            found = True
            if transform.get('format') != 'tfm':
                transform['format'] = 'tfm'
                needs_update = True
            break
    
    # If not found, add the raw_to_ortho transform entry
    if not found:
        T_set_metadata.append({
            'format': 'tfm',
            'name': 'raw_to_ortho',
            'type': 'affine'
        })
        needs_update = True
    
    if needs_update:
        if overwrite:
            # Save the updated metadata back to transforms.json
            with open(os.path.join(test_slice_path, "transforms.json"), 'w') as f:
                json.dump(T_set_metadata, f, indent=2)
        else:
            print("Corrected metadata (Not overwriting):")
            pprint.pprint(T_set_metadata)
    else:
        print("Metadata is already correct")

def ViewByNeu3DViewer(zarr_dir):
    """
    Open Neu3DViewer to view image block.
    """
    sys.path.append('/home/xyy/code/SimpleVolumeViewer/')
    #sys.path.append('/home/xyy/code/py/vtk_test/')
    import neu3dviewer
    from neu3dviewer.img_block_viewer import GUIControl


    # construct description of objects
    # see help of Neu3DViewer for possible options
    cmd_obj_desc = {
        'img_path': zarr_dir,
    }
    #neu3dviewer.utils.debug_level = 2
    neu3dviewer.utils.debug_level = 4
    print("Viewing in Neu3DViewer")
    gui = GUIControl()
    gui.EasyObjectImporter(cmd_obj_desc)
    gui.Start()

def BootstrappingResampling(test_slice_path, T_raw_to_ortho):
    print("Bootstrapping resampling tests.")
    target_voxel_size = _a((1.0, 1.0, 1.0))

    z_arr_raw = zarr.open(os.path.join(test_slice_path, '..', '..', '..', "visor_raw_images", "slice_1_10x.zarr", "0"), mode='r')
    print("Raw zarr array shape:", z_arr_raw.shape)

    if 0:
        # TODO: make the path in accord with std
        target_slice_path = os.path.join(test_slice_path, '..', '..', '..', "visor_recon_images", "xxx_20250627.zarr")

        z_arr = zarr.create_array(target_slice_path, shape=(100, 100, 100), chunks=(100, 100, 100), dtype='uint16', overwrite=True)
        z_arr[:,:,:] = np.random.randint(0, 256, size=(100, 100, 100), dtype='uint16')

    if 0:
        p = os.path.abspath(z_arr.store.root)
        #print(p)
        #p = '/home/xyy/code/visor-py/visor/tests/data/VISOR001.vsr/visor_raw_images/slice_1_10x.zarr/0'
        ViewByNeu3DViewer(p)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test transform/resample functionality")
    parser.add_argument("--bootstrap", type=int, choices=[0, 1], 
                       help="Run bootstrapping test (0=False, 1=True)")
    args = parser.parse_args()

    test_slice_path = "./data/VISOR001.vsr/visor_recon_transforms/xxx_20250525/slice_1_10x"
    os.makedirs(test_slice_path, exist_ok=True)
    
    if args.bootstrap is not None:
        print("Bootstrapping test for transform/resample")
        T_raw_to_stack = BootstrappingTransform(test_slice_path)
        pos_raw_set, pos_stack_set = GenerateTestData(T_raw_to_stack)
        v0 = [1, 0.5, 0.1]
        v1 = T_raw_to_stack.inverse().apply(T_raw_to_stack.apply(v0))
        npt.assert_allclose(v1, v0, rtol=1e-13, atol=2e-10)
        if args.bootstrap:
            # test against old code
            CompareWithOldCode(pos_raw_set, pos_stack_set)
        
        BootstrappingResampling(test_slice_path, T_raw_to_stack)
    else:
        print("No action specified. Use --bootstrap 0 or --bootstrap 1")