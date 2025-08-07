# Run test at root directory with below:
#   python -m unittest visor/tests/test_roi.py

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
        self.image_path = Path(__file__).parent/'data'/'VISOR001.vsr'/'visor_raw_images'/'slice_1_10x.zarr'
        self.resolution = '0'
        self.ranges = (1,1,slice(2,3),slice(None),slice(None))


class TestROI(TestBase):

    def setUp(self):
        super().setUp()

    def test_init(self):
        roi = visor.ROI(
            self.image_path,
            resolution=self.resolution,
            ranges=self.ranges,
        )
        self.assertIsInstance(roi, visor.ROI)
        self.assertIsInstance(roi.img, visor.Image)
        self.assertEqual(roi.resolution, self.resolution)
        self.assertEqual(roi.ranges, self.ranges)


class TestROILoad(TestBase):

    def setUp(self):
        super().setUp()
        self.roi = visor.ROI(
            self.image_path,
            resolution=self.resolution,
            ranges=self.ranges,
        )

    def test_load(self):
        np_arr = self.roi.load()
        self.assertIsInstance(np_arr, numpy.ndarray)
        self.assertEqual(np_arr.ndim, 3)
        self.assertEqual(np_arr.shape, (1, 4, 4))


if __name__ == '__main__':
    unittest.main()
