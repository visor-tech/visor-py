# Run test at root directory with below:
#   python -m unittest visor/tests/test_vsr.py

from pathlib import Path
import unittest
import shutil
import visor

class TestBase(unittest.TestCase):

    def setUp(self):
        # Test Data: validated with https://ome.github.io/ome-ngff-validator
        # simply use ome_zarr command line tool as below:
        #   $ ome_zarr view visor/image/tests/data/VISOR001/visor_raw_images/slice_1_10x.zarr
        self.path = Path(__file__).parent/'data'/'VISOR001.vsr'


class TestVSR(TestBase):

    def setUp(self):
        super().setUp()
        self.another_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'

    def tearDown(self):
        if self.another_vsr_path.exists():
            shutil.rmtree(self.another_vsr_path)

    def test_info(self):
        info = visor.info(self.path)
        self.assertEqual(info['animal_id'], 'VISOR001')
        self.assertEqual(info['project_name'], 'VISOR')
        self.assertEqual(info['species'], 'Mouse')
        self.assertEqual(info['subproject_name'], 'XXX-XXXX-1X7-3X')
        self.assertEqual(info['image_types'], ['raw','compr'])
        self.assertEqual(info['transform_versions'], ['xxx_20250525'])

    def test_info_not_vsr(self):
        not_vsr_path = Path(str(self.path).replace('.vsr',''))
        with self.assertRaises(ValueError) as context:
            visor.info(not_vsr_path)
        self.assertEqual(str(context.exception), f'The path {not_vsr_path} does not have .vsr extension.')

    def test_info_not_exist(self):
        with self.assertRaises(NotADirectoryError) as context:
            visor.info(self.another_vsr_path)
        self.assertEqual(str(context.exception), f'The path {self.another_vsr_path} is not a directory.')

    def test_list_image(self):
        images = visor.list_image(self.path)
        self.assertEqual(images, 
            {
                'raw': [
                    {
                        'name':'slice_1_10x',
                        'channels':['488','561'],
                        'resolutions':{
                            "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                        }
                    },
                    {
                        'name':'slice_1_10x_1',
                        'channels':['405','640'],
                        'resolutions':{
                            "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                        }
                    }
                ],
                'compr': [
                    {
                        'name':'xxx_slice_1_10x_20241201',
                        'channels':['405','640'],
                        'resolutions':{
                            "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                        }
                    }
                ]
            }
        )

    def test_list_image_by_type_raw(self):
        images = visor.list_image(self.path, type='raw')
        self.assertEqual(images, [
            {
                'name':'slice_1_10x',
                'channels':['488','561'],
                'resolutions':{
                    "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                }
            },
            {
                'name':'slice_1_10x_1',
                'channels':['405','640'],
                'resolutions':{
                    "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                }
            }
        ])

    def test_list_image_by_type_compr(self):
        images = visor.list_image(self.path, type='compr')
        self.assertEqual(images, [
            {
                'name':'xxx_slice_1_10x_20241201',
                'channels':['405','640'],
                'resolutions':{
                    "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                }
            }
        ])

    def test_list_image_by_channel_488(self):
        images = visor.list_image(self.path, channel='488')
        self.assertEqual(images,
            {
                'raw': [
                    {
                        'name':'slice_1_10x',
                        'channels':['488','561'],
                        'resolutions':{
                            "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                        }
                    },
                ],
                'compr': []
            }
        )

    def test_list_image_raw_by_type_and_channel(self):
        images = visor.list_image(self.path, type='raw', channel='488')
        self.assertEqual(images, [
            {
                'name':'slice_1_10x',
                'channels':['488','561'],
                'resolutions':{
                    "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                }
            }
        ])

    def test_list_transform(self):
        transforms = visor.list_transform(self.path)
        self.assertEqual(transforms,
            {
                'xxx_20250525': {
                    "spaces": ["raw","ortho","brain"],
                    "slices": [
                        {
                            "name": "slice_1_10x",
                            "transforms": ["raw_to_ortho","raw_to_brain"]
                        }
                    ]
                }
            }
        )

    def test_list_transform_by_version(self):
        transforms = visor.list_transform(self.path, version='xxx_20250525')
        self.assertEqual(transforms,
            {
                "spaces": ["raw","ortho","brain"],
                "slices": [
                    {
                        "name": "slice_1_10x",
                        "transforms": ["raw_to_ortho","raw_to_brain"]
                    }
                ]
            }
        )


if __name__ == '__main__':
    unittest.main()
