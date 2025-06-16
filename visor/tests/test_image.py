# Run test at root directory with below:
#   python -m unittest visor/tests/test_image.py

from pathlib import Path
import unittest
import shutil
import visor
import zarr
import numpy
import dask.array as da

class TestBase(unittest.TestCase):

    def setUp(self):
        self.path = Path(__file__).parent/'data'/'VISOR001.vsr'


class TestImage(TestBase):

    def setUp(self):
        super().setUp()
        self.another_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'

    def tearDown(self):
        if self.another_vsr_path.exists():
            shutil.rmtree(self.another_vsr_path)

    def test_load_image(self):
        arr = visor.load_image(self.path,
            type='raw',
            name='slice_1_10x',
            resolution='0'
        )
        self.assertIsInstance(arr, zarr.Array)
        self.assertEqual(arr.ndim, 5)
        self.assertEqual(arr.dtype, 'uint16')
        self.assertEqual(arr.shape, (2, 2, 4, 4, 4))
        self.assertEqual(arr.chunks, (1, 1, 2, 2, 2))
        self.assertEqual(arr.shards, (1, 1, 4, 4, 4))
        arr_compressor_info = arr.compressors[0].to_dict()['configuration']
        self.assertEqual(arr_compressor_info['cname'], 'zstd')
        self.assertEqual(arr_compressor_info['clevel'], 5)

        np_arr = arr[:]
        self.assertIsInstance(np_arr, numpy.ndarray)

        da_arr = da.from_array(arr, chunks=arr.chunks)
        self.assertIsInstance(da_arr, da.Array)


if __name__ == '__main__':
    unittest.main()
