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
        self.vsr_path = Path(__file__).parent/'data'/'VISOR001.vsr'
        self.image_type = 'raw'
        self.image_name = 'slice_1_10x'
        self.image_path = self.vsr_path/f'visor_{self.image_type}_images'/f'{self.image_name}.zarr'
        self.another_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'
        self.another_image_path = self.another_vsr_path/f'visor_{self.image_type}_images'/f'{self.image_name}.zarr'


class TestImage(TestBase):

    def setUp(self):
        super().setUp()

    def tearDown(self):
        if self.another_vsr_path.exists():
            shutil.rmtree(self.another_vsr_path)

    def test_init(self):
        img = visor.Image(
            self.vsr_path,
            image_type=self.image_type,
            image_name=self.image_name,
        )
        self.assertIsInstance(img, visor.Image)
        self.assertEqual(img.path, self.image_path)
        self.assertIsInstance(img.zgroup, zarr.Group)
        self.assertIn('ome', img.attrs)
        self.assertIn('visor', img.attrs)
        self.assertIn('visor_stacks', img.attrs['visor'])
        self.assertIn('channels', img.attrs['visor'])


class TestImageRead(TestBase):

    def setUp(self):
        super().setUp()
        self.img = visor.Image(
            self.vsr_path,
            image_type=self.image_type,
            image_name=self.image_name,
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
            self.vsr_path,
            image_type=self.image_type,
            image_name=self.image_name,
        )
        self.attrs = img_base.attrs

        self.img = visor.Image(
            self.another_vsr_path,
            image_type=self.image_type,
            image_name=self.image_name,
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
