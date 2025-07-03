# Run test at root directory with below:
#   python -m unittest visor/tests/test_transform.py

from pathlib import Path
import json
import unittest
import shutil
import visor
import SimpleITK as sitk

class TestBase(unittest.TestCase):

    def setUp(self):
        self.vsr_path = Path(__file__).parent/'data'/'VISOR001.vsr'
        self.recon_version = 'xxx_20250525'
        self.slice_name = 'slice_1_10x'
        self.another_slice_name = 'slice_2_10x'
        self.from_space = 'raw'
        self.to_space = 'ortho'
        recon_path = self.vsr_path/'visor_recon_transforms'/self.recon_version
        self.transform_path = recon_path/self.slice_name
        self.another_transform_path = recon_path/self.another_slice_name


class TestTransform(TestBase):

    def setUp(self):
        super().setUp()

    def tearDown(self):
        if self.another_transform_path.exists():
            shutil.rmtree(self.another_transform_path)

    def test_init(self):
        xfm = visor.Transform(
            self.vsr_path,
            recon_version=self.recon_version,
            slice_name=self.slice_name,
        )
        self.assertIsInstance(xfm, visor.Transform)
        self.assertEqual(xfm.path, self.transform_path)

    def test_init_not_exist(self):
        with self.assertRaises(NotADirectoryError) as context:
            visor.Transform(
                self.vsr_path,
                recon_version=self.recon_version,
                slice_name=self.another_slice_name,
            )
        self.assertEqual(str(context.exception),
                         f'The path {self.another_transform_path} is not a directory.')

    def test_create(self):
        xfm = visor.Transform(
            self.vsr_path,
            recon_version=self.recon_version,
            slice_name=self.another_slice_name,
            create=True,
        )
        self.assertIsInstance(xfm, visor.Transform)
        self.assertEqual(xfm.path, self.another_transform_path)

    def test_update_meta(self):

        trans_meta = {
            'name'  : 'raw_to_ortho',
            'type'  : 'affine',
            'format': 'tfm',
        }
        xfm = visor.Transform(
            self.vsr_path,
            recon_version=self.recon_version,
            slice_name=self.another_slice_name,
            create=True,
        )
        xfm.update_meta(trans = trans_meta)
        with open(xfm.path/'transforms.json') as trans_json:
            meta = json.load(trans_json)
        self.assertEqual(meta, trans_meta)


class TestTransformLoad(TestBase):

    def setUp(self):
        super().setUp()
        self.xfm = visor.Transform(
            self.vsr_path,
            recon_version=self.recon_version,
            slice_name=self.slice_name,
        )

    def test_load(self):
        t_raw_to_ortho = self.xfm.load(
            from_space='raw',
            to_space='ortho',
        )
        self.assertIsInstance(t_raw_to_ortho, sitk.Transform)
