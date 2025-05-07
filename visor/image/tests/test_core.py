# Run at root dir by
#   python -m unittest visor/image/tests/test_core.py

from pathlib import Path
import unittest
import shutil
import zarr
import numpy as np
import visor.image as vimg

class TestBase(unittest.TestCase):

    def setUp(self):

        # Test Data: validated with https://ome.github.io/ome-ngff-validator
        # simply use ome_zarr command line tool as below:
        #   $ ome_zarr view visor/image/tests/data/VISOR001/visor_raw_images/slice_1_10x.zarr
        self.path = Path(__file__).parent/'data'/'VISOR001.vsr'


class TestCore(TestBase):

    def test_open_vsr_r(self):

        img = vimg.open_vsr(self.path, 'r')
        
        self.assertIsInstance(img, vimg.core.Image)
        self.assertEqual(img.info['animal_id'], 'VISOR001')
        self.assertEqual(img.info['project_name'], 'VISOR')
        self.assertEqual(img.info['species'], 'Mouse')
        self.assertEqual(img.info['subproject_name'], 'XXX-XXXX-1X7-3X')
        self.assertEqual(img.image_types, ['raw','compr'])
        self.assertEqual(img.transforms, [])
        self.assertEqual(img.image_files, {
            'raw': [
                {'path':'slice_1_10x.zarr', 'channels':["488","561"]},
                {'path':'slice_1_10x_1.zarr', 'channels':["405","640"]}
            ],
            'compr': [
                {'path': 'xxx_slice_1_10x_20241201.zarr', 'channels': ['405', '640']}
            ]
        })


    def test_open_vsr_w(self):

        img = vimg.open_vsr(self.path, 'w')
        self.assertIsInstance(img, vimg.core.Image)


class TestImageReadOnly(TestBase):

    def setUp(self):

        super().setUp()
        self.img = vimg.open_vsr(self.path, 'r')

        self.zarr_file = 'slice_1_10x.zarr'
        self.resolution = 0


    def test_image_list(self):

        self.assertEqual(self.img.list(), {
            'raw': [
                {'path':'slice_1_10x.zarr', 'channels':["488","561"]},
                {'path':'slice_1_10x_1.zarr', 'channels':["405","640"]},
            ],
            'compr': [
                {'path': 'xxx_slice_1_10x_20241201.zarr', 'channels': ['405', '640']}
            ]
        })
        self.assertEqual(self.img.list('raw'), [
            {'path':'slice_1_10x.zarr', 'channels':["488","561"]},
            {'path':'slice_1_10x_1.zarr', 'channels':["405","640"]},
        ])
        self.assertEqual(self.img.list('projn'), {})


    def test_image_read(self):

        arr = self.img.read('raw',self.zarr_file,0)

        self.assertIsInstance(arr, vimg.core.Array)
        self.assertEqual(arr.info['zarr_format'], 3)
        self.assertIn('attributes', arr.info)
        self.assertIn('ome', arr.info['attributes'])
        self.assertIn('visor', arr.info['attributes'])
        self.assertEqual(arr.info['attributes']['ome']['version'], '0.5')
        self.assertIn('multiscales', arr.info['attributes']['ome'])
        self.assertIn('visor_stacks', arr.info['attributes']['visor'])
        self.assertIn('channels', arr.info['attributes']['visor'])
        self.assertEqual(arr.channel_map, {'488':0, '561':1})
        self.assertEqual(arr.stack_map, {'stack_1':0, 'stack_2':1})
        self.assertEqual(arr.array.shape, (2,2,4,4,4))

    def test_image_write_exception(self):

        arr = self.img.read('raw',self.zarr_file,resolution=self.resolution)
        with self.assertRaises(PermissionError) as context:
            self.img.write(arr,'raw',self.zarr_file,0,{},{},(),())
        self.assertEqual(str(context.exception), '"write" method is only available in "w" mode.')


class TestArrayRead(TestBase):

    def setUp(self):

        super().setUp()
        img = vimg.open_vsr(self.path, 'w')
        self.arr = img.read('raw','slice_1_10x.zarr',0)


    def test_array_read(self):

        arr = self.arr.read()
        self.assertEqual(arr.shape, (2,2,4,4,4))


    def test_array_read_channel(self):

        arr = self.arr.read(channel='488')
        self.assertEqual(arr.shape, (2,1,4,4,4))


    def test_array_read_stack(self):

        arr = self.arr.read(stack='stack_1')
        self.assertEqual(arr.shape, (1,2,4,4,4))


    def test_array_read_channel_and_stack(self):

        arr = self.arr.read(channel='488',stack='stack_1')
        self.assertEqual(arr.shape, (1,1,4,4,4))


    def test_array_read_channel_not_exist(self):

        with self.assertRaises(KeyError) as context:
            self.arr.read(channel='777')
        self.assertEqual(str(context.exception), '"channel 777 does not exist, valid channels: [\'488\', \'561\']"')


class TestImageWrite(TestBase):

    def setUp(self):

        super().setUp()
        self.src_img_type = 'raw'
        self.src_img_file = 'slice_1_10x'

        self.dst_img_type = 'raw'
        self.dst_img_file = 'slice_1_10x'
        
        self.resolution = 0

        src_img = vimg.open_vsr(self.path, 'r')
        self.src_img_info = src_img.info
        src_arr = src_img.read(self.src_img_type,
                                f'{self.src_img_file}.zarr',
                                self.resolution)
        self.arr = np.random.randint(0,255,size=(2,2,4,4,4), dtype='uint16')
        self.arr_info = src_arr.info['attributes']
        
        self.dst_path = self.path.parent/'tmp.vsr'


    def tearDown(self):

        if self.dst_path.exists():
            shutil.rmtree(self.dst_path)


    def test_image_write(self):
        
        dst_img = vimg.open_vsr(self.dst_path, 'w')
        dst_img.write(self.arr,
                      self.dst_img_type,
                      self.dst_img_file,
                      self.resolution,
                      self.src_img_info,
                      self.arr_info,
                      chunk_size=(1,1,2,2,2),
                      shard_size=(1,1,4,4,4),
                      selected=[{'path':'slice_1_10x.zarr','channels':['640']}])
        arr = dst_img.read(self.dst_img_type,
                           f'{self.dst_img_file}.zarr',
                           0)
        self.assertIsInstance(arr, vimg.core.Array)
        self.assertEqual(arr.info['zarr_format'], 3)
        self.assertIn('attributes', arr.info)
        self.assertIn('ome', arr.info['attributes'])
        self.assertIn('visor', arr.info['attributes'])
        self.assertEqual(arr.info['attributes']['ome']['version'], '0.5')
        self.assertIn('multiscales', arr.info['attributes']['ome'])
        self.assertIn('visor_stacks', arr.info['attributes']['visor'])
        self.assertIn('channels', arr.info['attributes']['visor'])
        self.assertEqual(arr.channel_map, {'488':0, '561':1})
        self.assertEqual(arr.stack_map, {'stack_1':0, 'stack_2':1})
        self.assertEqual(arr.array.shape, (2,2,4,4,4))

    def test_image_write_compress(self):
        dst_img = vimg.open_vsr(self.dst_path, 'w')
        compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=5)
        dst_img.write(self.arr,
                      self.dst_img_type,
                      self.dst_img_file,
                      self.resolution,
                      self.src_img_info,
                      self.arr_info,
                      chunk_size=(1,1,2,2,2),
                      shard_size=(1,1,4,4,4),
                      selected=[{'path':'slice_1_10x.zarr','channels':['640']}],
                      compressor=compressor)
        arr = dst_img.read(self.dst_img_type,
                           f'{self.dst_img_file}.zarr',
                           0)
        self.assertEqual((arr.array - self.arr).any(), False)

# class TestImageUpdate(TestBase):

#     def setUp(self):

#         super().setUp()
#         self.src_img_type = 'raw'
#         self.src_img_file = 'slice_1_10x'

#         self.dst_img_type = 'icorr'
#         self.dst_img_file = 'slice_1_10x'
        
#         self.resolution = 0

#         src_img = vimg.open_vsr(self.path, 'r')
#         self.src_img_info = src_img.info
#         self.arr = src_img.read(self.src_img_type,
#                                 f'{self.src_img_file}.zarr',
#                                 self.resolution)
        
#         self.dst_path = self.path/'visor_icorr_images'


#     def tearDown(self):

#         if self.dst_path.exists():
#             shutil.rmtree(self.dst_path)


#     def test_image_write(self):
        
#         dst_img = vimg.open_vsr(self.dst_path, 'w')
#         dst_img.write(self.arr.array,
#                       self.dst_img_type,
#                       self.dst_img_file,
#                       self.resolution,
#                       self.src_img_info,
#                       self.arr.info)
#         arr = dst_img.read(self.dst_img_type,
#                            f'{self.dst_img_file}.zarr',
#                            0)
#         self.assertIsInstance(arr, vimg.core.Array)
#         self.assertIn('multiscales', arr.info)
#         self.assertIn('visor_stacks', arr.info)
#         self.assertIn('channels', arr.info)
#         self.assertEqual(arr.channel_map, {'488':0, '561':1})
#         self.assertEqual(arr.stack_map, {'stack_1':0, 'stack_2':1})
#         self.assertEqual(arr.array.shape, (2,2,4,4,4))


if __name__ == '__main__':
    unittest.main()