# Run test at root directory with below:
#   python -m unittest visor/tests/test_transform.py

from pathlib import Path
import unittest
import shutil
import visor
import zarr
import numpy
import dask.array as da
from zarr.codecs import BloscCodec

class TestBase(unittest.TestCase):

    def setUp(self):
        self.vsr_path = Path(__file__).parent/'data'/'VISOR001.vsr'
        self.recon_version = 'xxx_20250525'
        self.slice_name = 'slice_1_10x',
        self.transform_name = 'raw_to_ortho',
        self.transform_path = self.vsr_path/'visor_recon_transforms'/self.recon_version/self.slice_name/self.transform_name
        # self.another_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'
        # self.another_image_path = self.another_vsr_path/f'visor_{self.image_type}_images'/f'{self.image_name}.zarr'


class TestTransform(TestBase):

    def setUp(self):
        super().setUp()

    # def tearDown(self):
    #     if self.another_vsr_path.exists():
    #         shutil.rmtree(self.another_vsr_path)

    def test_init(self):
        xfm = visor.Transform(
            self.vsr_path,
            recon_version=self.recon_version,
            slice_name=self.slice_name,
            transform_name=self.transform_name,
        )
        self.assertIsInstance(xfm, visor.Image)
        self.assertEqual(xfm.path, self.transform_path)
        self.assertEqual(xfm.type, 'affine')
        self.assertEqual(xfm.format, 'zarr')

# class TestSpaceList(TestBase):

# class TestTransformRead(TestBase):

#class TestTransformWrite(TestBase):

#class TestResample(TestBase):
    # resample(image, transform) -> image

#class TestTransformPoint(TestBase):
    # transform_point((coord, space), space) -> coord
