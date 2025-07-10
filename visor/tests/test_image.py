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
        self.another_image_name = 'slice_2_10x'
        self.another_image_path = self.vsr_path/f'visor_{self.image_type}_images'/f'{self.another_image_name}.zarr'


class TestImage(TestBase):

    def setUp(self):
        super().setUp()

    def tearDown(self):
        if self.another_image_path.exists():
            shutil.rmtree(self.another_image_path)

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

    def test_init_not_exist(self):
        with self.assertRaises(NotADirectoryError) as context:
            visor.Image(
                self.vsr_path,
                image_type=self.image_type,
                image_name=self.another_image_name,
            )
        self.assertEqual(str(context.exception),
                         f'The path {self.another_image_path} is not a directory.')

    def test_create(self):
        img = visor.Image(
            self.vsr_path,
            image_type=self.image_type,
            image_name=self.another_image_name,
            create=True,
        )
        self.assertIsInstance(img, visor.Image)
        self.assertEqual(img.path, self.another_image_path)


class TestImageLoad(TestBase):

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

    def test_load(self):
        arr = self.img.load(resolution='0')
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


class TestImageSave(TestBase):

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
            self.vsr_path,
            image_type=self.image_type,
            image_name=self.another_image_name,
            create=True,
        )

        self.zero_arr = numpy.zeros(
            self.new_arr_shape,
            dtype=self.dtype,
        )

        self.random_partial_arr = numpy.random.randint(
            0, 255,
            size=[1,2,4,4,4],
            dtype=self.dtype,
        )

    def tearDown(self):
        if self.another_image_path.exists():
            shutil.rmtree(self.another_image_path)

    def test_save(self):

        self.img.save(
            self.zero_arr,
            resolution='0',
            dtype=self.dtype,
            shape=self.new_arr_shape,
            shard_size=self.new_arr_shard_size,
            chunk_size=self.new_arr_chunk_size,
            compressors=BloscCodec(cname="zstd", clevel=5),
        )
        
        arr = self.img.load(resolution='0')
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

        self.img.save(
            self.zero_arr,
            resolution='0',
            dtype=self.dtype,
            shape=self.new_arr_shape,
            shard_size=self.new_arr_shard_size,
            chunk_size=self.new_arr_chunk_size,
            compressors=BloscCodec(cname="zstd", clevel=5),
        )
        
        arr = self.img.load(resolution='0')
        arr[:1,:,:,:,:] = self.random_partial_arr

        updated_arr = self.img.load(resolution='0')
        self.assertIsInstance(updated_arr, zarr.Array)
        updated_part = arr[:1,:,:,:,:]
        not_updated_part = arr[1:2,:,:,:,:]
        self.assertNotEqual(updated_part.sum(), 0) # should be updated
        self.assertEqual(not_updated_part.sum(), 0) # should not be updated


if __name__ == '__main__':
    unittest.main()
