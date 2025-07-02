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
        self.vsr_path = Path(__file__).parent/'data'/'VISOR001.vsr'


class TestVSR(TestBase):

    def setUp(self):
        super().setUp()
        self.another_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'

    def tearDown(self):
        if self.another_vsr_path.exists():
            shutil.rmtree(self.another_vsr_path)

    def test_init(self):
        vsr = visor.VSR(self.vsr_path)
        self.assertIsInstance(vsr, visor.VSR)

    def test_init_not_vsr(self):
        not_vsr_path = Path(str(self.vsr_path).replace('.vsr',''))
        with self.assertRaises(ValueError) as context:
            visor.VSR(not_vsr_path)
        self.assertEqual(str(context.exception), f'The path {not_vsr_path} does not have .vsr extension.')

    def test_init_not_exist(self):
        with self.assertRaises(NotADirectoryError) as context:
            visor.VSR(self.another_vsr_path)
        self.assertEqual(str(context.exception), f'The path {self.another_vsr_path} is not a directory.')

    def test_create_vsr(self):
        visor.VSR(self.another_vsr_path, create=True)
        self.assertTrue(self.another_vsr_path.is_dir())


class TestVSRMethods(TestBase):

    def setUp(self):
        super().setUp()
        self.vsr = visor.VSR(self.vsr_path)
        self.another_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'

    def tearDown(self):
        if self.another_vsr_path.exists():
            shutil.rmtree(self.another_vsr_path)

    def test_info(self):
        info = self.vsr.info()
        self.assertEqual(info['animal_id'], 'VISOR001')
        self.assertEqual(info['project_name'], 'VISOR')
        self.assertEqual(info['species'], 'Mouse')
        self.assertEqual(info['subproject_name'], 'XXX-XXXX-1X7-3X')
        self.assertEqual(info['image_types'], ['raw','compr'])
        self.assertEqual(info['recon_versions'], ['xxx_20250525'])

    def test_images(self):
        images = self.vsr.images()
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

    def test_images_by_type_raw(self):
        images = self.vsr.images(image_type='raw')
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

    def test_images_by_type_compr(self):
        images = self.vsr.images(image_type='compr')
        self.assertEqual(images, [
            {
                'name':'xxx_slice_1_10x_20241201',
                'channels':['405','640'],
                'resolutions':{
                    "0": [1.0, 1.0, 1.0, 1.0, 1.0]
                }
            }
        ])

    def test_transforms(self):
        transforms = self.vsr.transforms()
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

    def test_transforms_by_version(self):
        transforms = self.vsr.transforms(recon_version='xxx_20250525')
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
