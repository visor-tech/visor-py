# Run test at root directory with below:
#   python -m unittest visor/tests/test_transform.py

from pathlib import Path
import json
import unittest
import shutil
import visor
import SimpleITK as sitk
import numpy as np

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
        self.stack_idx = 0
        self.channel_idx = 0

    def test_load_affine_tfm(self):
        t_raw_to_ortho = self.xfm.load(
            from_space='raw',
            to_space='ortho',
            params=[self.stack_idx, self.channel_idx],
        )
        self.assertIsInstance(t_raw_to_ortho, sitk.Transform)
        self.assertEqual(sitk.Transform.GetParameters(t_raw_to_ortho),
                         (0.0, 0.876430630270142, 0.0,
                          0.0, 0.0, 1.03,
                          3.5, 0.5410816484822616, 0.0,
                          0.0, 0.0, 0.0))

    def test_load_affine_tfm_with_incorrect_params(self):
        with self.assertRaises(ValueError) as context:
            self.xfm.load(
                from_space='raw',
                to_space='ortho',
                params=[],
            )
        self.assertEqual(str(context.exception),
                         'Loading affine transform requires [stack_index, channel_index] in params.')

    def test_load_not_exist(self):
        with self.assertRaises(FileNotFoundError) as context:
            self.xfm.load(
                from_space='raw',
                to_space='brain',
                params=[self.stack_idx, self.channel_idx],
            )
        self.assertEqual(str(context.exception),
                         f'Transform raw_to_brain is not in {self.transform_path}.')


class TestTransformSave(TestBase):

    def setUp(self):
        super().setUp()
        self.xfm = visor.Transform(
            self.vsr_path,
            recon_version=self.recon_version,
            slice_name=self.another_slice_name,
            create=True,
        )

        self.stack_idx = 0
        self.channel_idx = 0
        self.affine_mat = [ 0, np.sin(45) * 1.03, 0,
                            0, 0, 1.03,
                            3.5, np.cos(45) * 1.03, 0 ]
        self.affine_vec = [0,0,0]
        self.params = [self.stack_idx] + [self.channel_idx] + self.affine_mat + self.affine_vec

    def tearDown(self):
        if self.another_transform_path.exists():
            shutil.rmtree(self.another_transform_path)

    def test_save_affine_tfm(self):
        t_type = 'affine'
        t_format = 'tfm'
        self.xfm.save(
            from_space=self.from_space,
            to_space=self.to_space,
            t_type=t_type,
            t_format=t_format,
            params=self.params,
        )

        space_path = self.another_transform_path/f'{self.from_space}_to_{self.to_space}'
        self.assertTrue(space_path.is_dir)
        self.assertTrue((space_path/str(self.stack_idx)/str(self.channel_idx)).is_dir)
        self.assertTrue((space_path/str(self.stack_idx)/str(self.channel_idx)/f'{t_type}.{t_format}').exists)

        self.xfm.update_meta(
            trans = [{
                'name'  : f'{self.from_space}_to_{self.to_space}',
                'type'  : t_type,
                'format': t_format,
            }]
        )

        t_raw_to_ortho = self.xfm.load(
            from_space=self.from_space,
            to_space=self.to_space,
            params=[self.stack_idx, self.channel_idx],
        )
        self.assertIsInstance(t_raw_to_ortho, sitk.Transform)
        self.assertEqual(sitk.Transform.GetParameters(t_raw_to_ortho),
                         (0.0, 0.876430630270142, 0.0,
                          0.0, 0.0, 1.03,
                          3.5, 0.5410816484822616, 0.0,
                          0.0, 0.0, 0.0))

    def test_save_affine_tfm_with_incorrect_params(self):
        with self.assertRaises(ValueError) as context:
            self.xfm.save(
                from_space=self.from_space,
                to_space=self.to_space,
                t_type='affine',
                t_format='tfm',
                params=[],
            )
        self.assertEqual(str(context.exception),
                         'Saving affine transform requires [stack_index, channel_index, affine_mat, affine_vec] in params.')
