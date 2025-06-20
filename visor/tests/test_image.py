# Run test at root directory with below:
#   python -m unittest visor/tests/test_image.py

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
        self.path = Path(__file__).parent/'data'/'VISOR001.vsr'


class TestImage(TestBase):

    def setUp(self):
        super().setUp()
        self.type = 'raw'
        self.name = 'slice_1_10x'
        self.image_path = self.path/f'visor_{self.type}_images'/f'{self.name}.zarr'
        self.another_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'
        self.another_image_path = self.another_vsr_path/f'visor_{self.type}_images'/f'{self.name}.zarr'

    def tearDown(self):
        if self.another_vsr_path.exists():
            shutil.rmtree(self.another_vsr_path)

    def test_init_read_only(self):
        v_img_r = visor.Image(
            self.path,
            type=self.type,
            name=self.name,
            mode='r',
        )
        self.assertIsInstance(v_img_r, visor.Image)
        self.assertEqual(v_img_r.path, self.image_path)
        self.assertEqual(v_img_r.mode, 'r')
        self.assertIsInstance(v_img_r.store, zarr.Group)

    def test_init_default_mode(self):
        v_img_r = visor.Image(
            self.path,
            type=self.type,
            name=self.name,
        )
        self.assertEqual(v_img_r.mode, 'r')

    def test_init_read_not_exist(self):
        with self.assertRaises(NotADirectoryError) as context:
            visor.Image(
                self.another_vsr_path,
                type=self.type,
                name=self.name,
            )
        self.assertEqual(str(context.exception), f'The image path {self.another_image_path} is not valid.')

    def test_init_invalid_mode(self):
        with self.assertRaises(ValueError) as context:
            visor.Image(
                self.path,
                type=self.type,
                name=self.name,
                mode='w',
            )
        self.assertEqual(str(context.exception), f'Invalid mode \'w\': mode values could be \'r\' or \'w-\'.')

    def test_init_create_only(self):

        attr = {}
        new_arr_shape      = (2,2,4,4,4)
        new_arr_shard_size = (1,1,4,4,4)
        new_arr_chunk_size = (1,1,2,2,2)

        v_img_c = visor.Image(
            self.another_vsr_path,
            type=self.type,
            name=self.name,
            mode='w-',
            attr=attr,
            resolution='0',
            dtype='uint16',
            shape=new_arr_shape,
            shard_size=new_arr_shard_size,
            chunk_size=new_arr_chunk_size,
            compressors=BloscCodec(cname="zstd", clevel=5),
        )
        self.assertIsInstance(v_img_c, visor.Image)
        self.assertEqual(v_img_c.path, self.another_image_path)
        self.assertEqual(v_img_c.mode, 'w-')
        self.assertIsInstance(v_img_c.store, zarr.Group)

    def test_init_create_but_exist(self):
        with self.assertRaises(FileExistsError) as context:
            visor.Image(
                self.path,
                type=self.type,
                name=self.name,
                mode='w-',
            )
        self.assertEqual(str(context.exception), f'The image path {self.image_path} already exist.')


    # def test_open_create_new(self):
    #     vsr = visor.open_vsr(self.new_vsr_path, 'w')
    #     self.assertIsInstance(vsr, visor.core.VSR)
    #     self.assertEqual(vsr.path, self.new_vsr_path)
    #     self.assertEqual(vsr.mode, 'w')

    # def test_load_image(self):
    #     arr = visor.load_image(self.path,
    #         type=self.type,
    #         name=self.name,
    #         resolution='0'
    #     )
    #     self.assertIsInstance(arr, zarr.Array)
    #     self.assertEqual(arr.ndim, 5)
    #     self.assertEqual(arr.dtype, 'uint16')
    #     self.assertEqual(arr.shape, (2, 2, 4, 4, 4))
    #     self.assertEqual(arr.chunks, (1, 1, 2, 2, 2))
    #     self.assertEqual(arr.shards, (1, 1, 4, 4, 4))
    #     arr_compressor_info = arr.compressors[0].to_dict()['configuration']
    #     self.assertEqual(arr_compressor_info['cname'], 'zstd')
    #     self.assertEqual(arr_compressor_info['clevel'], 5)

    #     np_arr = arr[:]
    #     self.assertIsInstance(np_arr, numpy.ndarray)

    #     da_arr = da.from_array(arr, chunks=arr.chunks)
    #     self.assertIsInstance(da_arr, da.Array)


if __name__ == '__main__':
    unittest.main()
