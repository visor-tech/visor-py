# Run test at root directory with below:
#   python -m unittest visor/tests/test_image.py

from pathlib import Path
import json
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
        self.type = 'raw'
        self.name = 'slice_1_10x'
        self.image_path = self.path/f'visor_{self.type}_images'/f'{self.name}.zarr'
        self.another_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'
        self.another_image_path = self.another_vsr_path/f'visor_{self.type}_images'/f'{self.name}.zarr'


class TestImage(TestBase):

    def setUp(self):
        super().setUp()

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
        self.assertIsInstance(v_img_r.zgroup, zarr.Group)
        self.assertIn('ome', v_img_r.attrs)
        self.assertIn('visor', v_img_r.attrs)
        self.assertIn('visor_stacks', v_img_r.attrs['visor'])
        self.assertIn('channels', v_img_r.attrs['visor'])

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

        v_img_c = visor.Image(
            self.another_vsr_path,
            type=self.type,
            name=self.name,
            mode='w-',
        )
        self.assertIsInstance(v_img_c, visor.Image)
        self.assertEqual(v_img_c.path, self.another_image_path)
        self.assertEqual(v_img_c.mode, 'w-')
        self.assertIsInstance(v_img_c.zgroup, zarr.Group)

    def test_init_create_but_exist(self):
        with self.assertRaises(FileExistsError) as context:
            visor.Image(
                self.path,
                type=self.type,
                name=self.name,
                mode='w-',
            )
        self.assertEqual(str(context.exception), f'The image path {self.image_path} already exist.')


class TestImageRead(TestBase):

    def setUp(self):
        super().setUp()
        self.img = visor.Image(
            self.path,
            type=self.type,
            name=self.name,
            mode='r',
        )

    def test_label_to_index(self):
        s1_idx = self.img.label_to_index('stack', 'stack_1')
        self.assertEqual(s1_idx, 0)

        c488_idx = self.img.label_to_index('channel', '488')
        self.assertEqual(c488_idx, 0)

    def test_open(self):
        arr = self.img.open(resolution='0')
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
        self.assertEqual(np_arr.ndim, 5)
        self.assertEqual(np_arr.shape, (2, 2, 4, 4, 4))
        sub_np_arr = arr[:1,:1,:,:,:]
        self.assertEqual(sub_np_arr.ndim, 5)
        self.assertEqual(sub_np_arr.shape, (1, 1, 4, 4, 4))

        da_arr = da.from_array(arr, chunks=arr.chunks)
        self.assertIsInstance(da_arr, da.Array)


class TestImageWrite(TestBase):

    def setUp(self):
        super().setUp()
        self.new_arr_shape      = (2,2,4,4,4)
        self.new_arr_shard_size = (1,1,4,4,4)
        self.new_arr_chunk_size = (1,1,2,2,2)
        self.dtype = 'uint16'

        img_base = visor.Image(
            self.path,
            type=self.type,
            name=self.name,
            mode='r',
        )
        self.attrs = img_base.attrs

        self.img = visor.Image(
            self.another_vsr_path,
            type=self.type,
            name=self.name,
            mode='w-',
        )

        self.new_arr = numpy.random.randint(
            0, 255,
            size=self.new_arr_shape,
            dtype=self.dtype,
        )

    def tearDown(self):
        if self.another_vsr_path.exists():
            shutil.rmtree(self.another_vsr_path)

    def test_create_and_save(self):
        zarray = self.img.create(
            resolution='0',
            dtype='uint16',
            attrs=self.attrs,
            shape=self.new_arr_shape,
            shard_size=self.new_arr_shard_size,
            chunk_size=self.new_arr_chunk_size,
            compressors=BloscCodec(cname="zstd", clevel=5),
        )

        zarray[...] = self.new_arr
        
        arr = self.img.open(resolution='0')
        self.assertIsInstance(arr, zarr.Array)
        self.assertEqual(arr.ndim, 5)
        self.assertEqual(arr.dtype, 'uint16')
        self.assertEqual(arr.shape, (2, 2, 4, 4, 4))
        self.assertEqual(arr.chunks, (1, 1, 2, 2, 2))
        self.assertEqual(arr.shards, (1, 1, 4, 4, 4))
        arr_compressor_info = arr.compressors[0].to_dict()['configuration']
        self.assertEqual(arr_compressor_info['cname'], 'zstd')
        self.assertEqual(arr_compressor_info['clevel'], 5)

    def test_save_partially(self):
        zarray = self.img.create(
            resolution='1',
            attrs=self.attrs,
            dtype='uint16',
            shape=self.new_arr_shape,
            shard_size=self.new_arr_shard_size,
            chunk_size=self.new_arr_chunk_size,
            compressors=BloscCodec(cname="zstd", clevel=5),
        )

        zarray[:1,:,:,:,:] = self.new_arr[:1,:,:,:,:]

        arr = self.img.open(resolution='1')
        self.assertIsInstance(arr, zarr.Array)
        sub_np_arr = arr[1:2,:,:,:,:] # no data yet
        self.assertEqual(sub_np_arr.ndim, 5)
        self.assertEqual(sub_np_arr.dtype, 'uint16')
        self.assertEqual(sub_np_arr.shape, (1, 2, 4, 4, 4))
        self.assertEqual(sub_np_arr.sum(), 0) # should be empty array

if __name__ == '__main__':
    unittest.main()
