# Run at root dir by
#   python -m unittest visor/tests/test_core.py

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


class TestOpen(TestBase):

    def setUp(self):
        self.new_vsr_path = Path(__file__).parent/'data'/'VISOR002.vsr'

    def tearDown(self):
        if self.new_vsr_path.exists():
            shutil.rmtree(self.new_vsr_path)

    def test_open_read_only(self):
        vsr = visor.open(self.path, 'r')
        self.assertIsInstance(vsr, visor.core.VSR)
        self.assertEqual(vsr.path, self.path)
        self.assertEqual(vsr.mode, 'r')

    def test_open_read_write(self):
        vsr = visor.open(self.path, 'w')
        self.assertIsInstance(vsr, visor.core.VSR)
        self.assertEqual(vsr.path, self.path)
        self.assertEqual(vsr.mode, 'w')

    def test_open_default_model(self):
        vsr = visor.open(self.path)
        self.assertEqual(vsr.mode, 'r')

    def test_open_not_vsr(self):
        with self.assertRaises(ValueError) as context:
            visor.open(self.path.replace('.vsr',''))
        self.assertEqual(str(context.exception), f'The path {self.path} does not have .vsr extension.')

    def test_open_not_exist(self):
        with self.assertRaises(NotADirectoryError) as context:
            visor.open(self.new_vsr_path)
        self.assertEqual(str(context.exception), f'The path {self.path} is not a directory.')

    def test_open_create_new(self):
        vsr = visor.open(self.new_vsr_path, 'w')
        self.assertIsInstance(vsr, visor.core.VSR)
        self.assertEqual(vsr.path, self.new_vsr_path)
        self.assertEqual(vsr.mode, 'w')


class TestInfo(TestBase):

    def setUp(self):
        super().setUp()
        self.vsr = visor.open(self.path, 'r')

    def test_get_info(self):

        info = self.vsr.get_info()

        self.assertEqual(info.animal_id, 'VISOR001')
        self.assertEqual(info.project_name, 'VISOR')
        self.assertEqual(info.species, 'Mouse')
        self.assertEqual(info.subproject_name, 'XXX-XXXX-1X7-3X')
        self.assertEqual(info.transform_versions, [])
        self.assertEqual(info.image_files, {
            'raw': [
                {'path':'slice_1_10x.zarr', 'channels':["488","561"],
                 'resolutions':[{
                     "path": "0",
                     "coordinateTransformations": [{
                         "type": "scale",
                         "scale": [1.0, 1.0, 1.0, 1.0, 1.0]
                     }]}
                 ]},
                {'path':'slice_1_10x_1.zarr', 'channels':["405","640"],
                 'resolutions':[{
                     "path": "0",
                     "coordinateTransformations": [{
                         "type": "scale",
                         "scale": [1.0, 1.0, 1.0, 1.0, 1.0]
                     }]}
                 ]}
            ],
            'compr': [
                {'path': 'xxx_slice_1_10x_20241201.zarr', 'channels': ['405', '640'],
                 'resolutions':[{
                     "path": "0",
                     "coordinateTransformations": [{
                         "type": "scale",
                         "scale": [1.0, 1.0, 1.0, 1.0, 1.0]
                     }]}
                 ]}
            ]
        })


if __name__ == '__main__':
    unittest.main()