# Tests for the 2026.9.1 transform additions:
#   multi-entry transforms.json loading (bug fix), inverse-direction loading
#   with declared direction metadata, ddf (mha) slice-level and per-channel
#   load/save, overwrite semantics, and save-from-transform-object.
#
# Run at root directory with:
#   python -m unittest visor/tests/test_transform_v2.py

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
        self.slice_name = 'slice_2_10x'
        recon_path = self.vsr_path/'visor_recon_transforms'/self.recon_version
        self.transform_path = recon_path/self.slice_name
        self.stack_idx = 0
        self.channel_idx = 0
        self.affine_mat = [0, np.sin(45) * 1.03, 0,
                           0, 0, 1.03,
                           3.5, np.cos(45) * 1.03, 0]
        self.affine_vec = [0, 0, 0]
        self.params = ([self.stack_idx] + [self.channel_idx]
                       + self.affine_mat + self.affine_vec)

    def tearDown(self):
        if self.transform_path.exists():
            shutil.rmtree(self.transform_path)

    def _xfm(self, create=True):
        return visor.Transform(
            self.vsr_path,
            recon_version=self.recon_version,
            slice_name=self.slice_name,
            create=create,
        )


class TestTransformV2(TestBase):

    def test_load_second_entry_in_meta(self):
        # regression: load used to raise on the first non-matching entry of
        # transforms.json, so any non-first entry failed to load
        xfm = self._xfm()
        xfm.save('raw', 'ortho', 'affine', 'tfm', params=self.params)
        xfm.save('raw', 'slice', 'affine', 'tfm', params=self.params)
        xfm.update_meta(trans=[
            {'name': 'raw_to_ortho', 'type': 'affine', 'format': 'tfm'},
            {'name': 'raw_to_slice', 'type': 'affine', 'format': 'tfm'},
        ])
        t = xfm.load('raw', 'slice', params=[self.stack_idx, self.channel_idx])
        self.assertIsInstance(t, sitk.Transform)

    def test_load_inverse_direction(self):
        xfm = self._xfm()
        t = sitk.AffineTransform(3)
        t.SetMatrix(self.affine_mat)
        t.SetTranslation((1.0, 2.0, 3.0))
        xfm.save('raw', 'ortho', 'affine', 'tfm', transform=t,
                 stack=self.stack_idx, channel=self.channel_idx)
        xfm.update_meta(trans=[
            {'name': 'raw_to_ortho', 'type': 'affine', 'format': 'tfm',
             'direction': 'raw_to_ortho'},
        ])

        forward = xfm.load('raw', 'ortho',
                           params=[self.stack_idx, self.channel_idx])
        self.assertAlmostEqual(forward.TransformPoint((0, 0, 0))[0], 1.0)

        backward = xfm.load('ortho', 'raw',
                            params=[self.stack_idx, self.channel_idx])
        self.assertIsInstance(backward, sitk.Transform)
        # inverse maps a forward-mapped point back to its origin
        mapped = forward.TransformPoint((2.0, 3.0, 4.0))
        unmapped = backward.TransformPoint(mapped)
        for a, b in zip(unmapped, (2.0, 3.0, 4.0)):
            self.assertAlmostEqual(a, b, places=5)

    def test_load_inverse_by_name_flipped_direction(self):
        # entry NAME says ortho_to_raw but the stored mapping direction is
        # declared raw_to_ortho: requesting raw_to_ortho returns it as-is,
        # requesting ortho_to_raw returns the inverse
        xfm = self._xfm()
        t = sitk.AffineTransform(3)
        t.Translate((5.0, 0.0, 0.0))
        xfm.save('ortho', 'raw', 'affine', 'tfm', transform=t,
                 stack=self.stack_idx, channel=self.channel_idx)
        xfm.update_meta(trans=[
            {'name': 'ortho_to_raw', 'type': 'affine', 'format': 'tfm',
             'direction': 'raw_to_ortho'},
        ])
        as_is = xfm.load('raw', 'ortho', params=[self.stack_idx, self.channel_idx])
        self.assertAlmostEqual(as_is.TransformPoint((5.0, 0.0, 0.0))[0], 10.0)
        inverted = xfm.load('ortho', 'raw', params=[self.stack_idx, self.channel_idx])
        self.assertAlmostEqual(inverted.TransformPoint((5.0, 0.0, 0.0))[0], 0.0)

    def test_load_invalid_direction_meta(self):
        xfm = self._xfm()
        xfm.save('raw', 'ortho', 'affine', 'tfm', params=self.params)
        xfm.update_meta(trans=[
            {'name': 'raw_to_ortho', 'type': 'affine', 'format': 'tfm',
             'direction': 'slice_to_brain'},
        ])
        with self.assertRaises(ValueError):
            xfm.load('raw', 'ortho', params=[self.stack_idx, self.channel_idx])


class TestTransformDDF(TestBase):

    def _field(self):
        arr = np.zeros((2, 4, 4, 3), np.float64)
        arr[..., 0] = 0.5
        arr[..., 1] = 0.25
        img = sitk.GetImageFromArray(arr, isVector=True)
        img.SetSpacing((4.0, 4.0, 300.0))
        img.SetOrigin((10.0, 20.0, 30.0))
        return img

    def test_save_load_slice_level(self):
        xfm = self._xfm()
        img = self._field()
        path = xfm.save('slice', 'sample', 'dense displacement field', 'mha',
                        transform=img)
        self.assertEqual(path.name, 'dense displacement field.mha')
        xfm.update_meta(trans=[
            {'name': 'slice_to_sample', 'type': 'dense displacement field',
             'format': 'mha'},
        ])

        t = xfm.load('slice', 'sample')
        self.assertIsInstance(t, sitk.DisplacementFieldTransform)
        field = sitk.Cast(t.GetDisplacementField(), sitk.sitkVectorFloat32)
        self.assertEqual(field.GetSize(), (4, 4, 2))
        self.assertEqual(field.GetSpacing(), (4.0, 4.0, 300.0))
        self.assertEqual(field.GetOrigin(), (10.0, 20.0, 30.0))
        # displacement is additive: p + disp
        self.assertAlmostEqual(t.TransformPoint((11.0, 21.0, 31.0))[0],
                               11.0 + 0.5, places=5)

    def test_save_load_per_channel(self):
        xfm = self._xfm()
        img = self._field()
        path = xfm.save('slice', 'sample', 'dense displacement field', 'mha',
                        transform=img, channel=1)
        self.assertEqual(path.name, 'dense displacement field.mha')
        self.assertEqual(path.parent.name, '1')
        xfm.update_meta(trans=[
            {'name': 'slice_to_sample', 'type': 'dense displacement field',
             'format': 'mha'},
        ])
        t = xfm.load('slice', 'sample', channel=1)
        self.assertIsInstance(t, sitk.DisplacementFieldTransform)

    def test_load_ddf_inverse_raises(self):
        xfm = self._xfm()
        xfm.save('slice', 'sample', 'dense displacement field', 'mha',
                 transform=self._field())
        xfm.update_meta(trans=[
            {'name': 'slice_to_sample', 'type': 'dense displacement field',
             'format': 'mha'},
        ])
        with self.assertRaises(RuntimeError):
            xfm.load('sample', 'slice')

    def test_save_ddf_requires_transform(self):
        xfm = self._xfm()
        with self.assertRaises(ValueError):
            xfm.save('slice', 'sample', 'dense displacement field', 'mha')

    def test_save_from_displacement_field_transform(self):
        xfm = self._xfm()
        img = self._field()
        t = sitk.DisplacementFieldTransform(
            sitk.Cast(img, sitk.sitkVectorFloat64))
        xfm.save('slice', 'sample', 'dense displacement field', 'mha',
                 transform=t)
        xfm.update_meta(trans=[
            {'name': 'slice_to_sample', 'type': 'dense displacement field',
             'format': 'mha'},
        ])
        back = xfm.load('slice', 'sample')
        self.assertAlmostEqual(
            back.TransformPoint((11.0, 21.0, 31.0))[0], 11.5, places=5)


class TestTransformOverwrite(TestBase):

    def test_save_refuses_existing(self):
        xfm = self._xfm()
        xfm.save('raw', 'ortho', 'affine', 'tfm', params=self.params)
        with self.assertRaises(FileExistsError):
            xfm.save('raw', 'ortho', 'affine', 'tfm', params=self.params)

    def test_save_overwrite(self):
        xfm = self._xfm()
        xfm.save('raw', 'ortho', 'affine', 'tfm', params=self.params)
        t = sitk.AffineTransform(3)
        t.Translate((7.0, 0.0, 0.0))
        xfm.save('raw', 'ortho', 'affine', 'tfm', transform=t,
                 stack=self.stack_idx, channel=self.channel_idx,
                 overwrite=True)
        xfm.update_meta(trans=[
            {'name': 'raw_to_ortho', 'type': 'affine', 'format': 'tfm'},
        ])
        got = xfm.load('raw', 'ortho', params=[self.stack_idx, self.channel_idx])
        self.assertAlmostEqual(got.TransformPoint((0, 0, 0))[0], 7.0)

    def test_save_from_transform_object_requires_indices(self):
        xfm = self._xfm()
        t = sitk.AffineTransform(3)
        with self.assertRaises(ValueError):
            xfm.save('raw', 'ortho', 'affine', 'tfm', transform=t)

    def test_update_meta_append(self):
        xfm = self._xfm()
        xfm.update_meta(trans=[
            {'name': 'raw_to_ortho', 'type': 'affine', 'format': 'tfm'},
        ])
        xfm.update_meta(
            trans={'name': 'slice_to_sample',
                   'type': 'dense displacement field', 'format': 'mha'},
            append=True)
        with open(xfm.path/'transforms.json') as f:
            meta = json.load(f)
        self.assertEqual(len(meta), 2)
        self.assertEqual(meta[1]['name'], 'slice_to_sample')


if __name__ == '__main__':
    unittest.main()
