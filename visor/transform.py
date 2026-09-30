from pathlib import Path
import json

from ._zarrs import enable_zarrs_acceleration
import SimpleITK as sitk

class Transform:

    def __init__(self, vsr_path:str|Path,
                 recon_version:str, slice_name:str, create=False):
        """
        Constructor of Transform

        Parameters:
            vsr_path:      path to the .vsr file
            recon_version: reconstruction version, see vsr.info()['recon_versions']
            slice_name:    slice directory name, see vsr.transforms()
            create:        boolean
        """
        enable_zarrs_acceleration()
        vsr_path = Path(vsr_path)
        # Validate vsr path
        if vsr_path.suffix != '.vsr':
            raise ValueError(f'The path {vsr_path} is not valid, must contain .vsr extension.')
        if not vsr_path.exists() or not vsr_path.is_dir():
            raise NotADirectoryError(f'The path {vsr_path} is not a directory.')

        transform_path = vsr_path/'visor_recon_transforms'/recon_version/slice_name
        if create:
            transform_path.mkdir(parents=True, exist_ok=True)
            if not (transform_path/'transforms.json').exists():
                with open(transform_path/'transforms.json', 'w') as trans_json:
                    trans_json.write('{\n  "_comment": "see https://visor-tech.github.io/visor-data-schema/"\n}')
        if not transform_path.exists() or not transform_path.is_dir():
            raise NotADirectoryError(f'The path {transform_path} is not a directory.')
        self.path = transform_path


    def load(self, from_space:str, to_space:str, params=None,
             stack:int=None, channel:int=None):
        """
        Load Transform

        The transform is returned as a point map ``T: from_space -> to_space``
        (i.e. ``p_to = T(p_from)``). If the stored transform has the opposite
        direction, it is inverted on load when an analytic inverse exists
        (affine); displacement-field transforms have no analytic inverse and
        raise instead -- store the needed direction explicitly.

        Parameters:
            from_space: source space name
            to_space:   target space name
            params:     parameters to identify the transform:
                        affine (per stack+channel): [stack_index, channel_index]
                        ddf (slice-level): [] or None
                        ddf (per channel): [channel_index]
            stack:      stack index (alternative to params)
            channel:    channel index (alternative to params)

        Return:
            SimpleITK.Transform
        """
        t_meta_file = self.path/'transforms.json'
        if not t_meta_file.exists():
            raise FileNotFoundError(f'Metadata file transforms.json is not found in {self.path}.')
        with open(t_meta_file) as f:
            t_list = json.load(f)

        t_name = f'{from_space}_to_{to_space}'
        t_inv_name = f'{to_space}_to_{from_space}'

        entry = None
        for t in t_list:
            if t.get('name') in (t_name, t_inv_name):
                entry = t
                break
        if entry is None:
            raise FileNotFoundError(f'Transform {t_name} is not in {self.path}.')

        # 'direction' declares the mapping direction of the stored transform
        # (which way the loaded point map goes); it defaults to the entry
        # name. The entry name only identifies the entry.
        direction = entry.get('direction', entry['name'])
        if direction == t_inv_name:
            invert = True
        elif direction == t_name:
            invert = False
        else:
            raise ValueError(
                f"Transform {entry['name']} declares invalid direction '{direction}' "
                f"(expected '{t_name}' or '{t_inv_name}').")

        transform = self._load_transform(
            entry=entry,
            params=params,
            stack=stack,
            channel=channel,
        )
        if invert:
            transform = self._invert(transform, entry)
        return transform


    def save(self, from_space:str, to_space:str,
             t_type:str, t_format:str, params=None, *,
             transform=None, stack:int=None, channel:int=None,
             overwrite=False):
        """
        Save Transform

        Storage layout (per slice directory):
            per stack+channel (affine):
                {from}_to_{to}/{stack}/{channel}/{type}.{format}
            slice-level (ddf):
                {from}_to_{to}/{type}.{format}
            per channel (ddf):
                {from}_to_{to}/{channel}/{type}.{format}

        Parameters:
            from_space: source space name
            to_space:   target space name
            t_type:     transform type: 'affine' | 'dense displacement field'
            t_format:   transform store format: 'tfm' | 'mha'
            params:     for affine (legacy form):
                        [stack_index, channel_index, affine_mat(9), affine_vec(3)]
            transform:  alternative to params: a SimpleITK.AffineTransform(3),
                        or a vector image / DisplacementFieldTransform for ddf
            stack:      stack index (alternative to params, affine)
            channel:    channel index (alternative to params)
            overwrite:  replace an existing transform directory if True

        Return:
            Path of the written transform file
        """
        t_name = f'{from_space}_to_{to_space}'
        t_path = self.path/t_name

        if 'affine' == t_type and 'tfm' == t_format:
            stack_idx, channel_idx, t_mat, t_vec = self._affine_params(
                params, transform, stack, channel)
            file_path = t_path/str(stack_idx)/str(channel_idx)/f'{t_type}.{t_format}'
            self._prepare_write(file_path, overwrite)
            t = sitk.AffineTransform(3)
            t.SetMatrix(t_mat)
            t.SetTranslation(t_vec)
            sitk.WriteTransform(t, file_path)
            return file_path

        if 'dense displacement field' == t_type and 'mha' == t_format:
            field = self._ddf_field(transform)
            if channel is not None:
                file_path = t_path/str(channel)/f'{t_type}.{t_format}'
            else:
                file_path = t_path/f'{t_type}.{t_format}'
            self._prepare_write(file_path, overwrite)
            # legacy-compatible storage: float32 vector image; grid
            # origin/spacing carry the physical placement
            sitk.WriteImage(sitk.Cast(field, sitk.sitkVectorFloat32), file_path)
            return file_path

        raise NotImplementedError(
            f"Transform type '{t_type}' with format '{t_format}' is not supported yet; "
            f"supported: affine+tfm, 'dense displacement field'+mha.")

    @staticmethod
    def _prepare_write(file_path: Path, overwrite: bool):
        """Create parent dirs; guard duplicates at file granularity (a
        directory holds many stacks/channels, so overwrite never removes
        sibling transform files)."""
        if file_path.exists():
            if overwrite:
                file_path.unlink()
            else:
                raise FileExistsError(f'The transform {file_path} already exists.')
        file_path.parent.mkdir(parents=True, exist_ok=True)


    def update_meta(self, recon:dict=None, trans:list|dict=None, append:bool=False):
        """
        Update transforms.json (and recon.json of the recon version)

        Parameters:
            recon:  new recon.json content (replaces the file)
            trans:  new transforms.json content (list of entries, or a single
                    dict when append=True)
            append: add trans entries to the existing list instead of replacing
        """
        if recon:
            recon_json = self.path.parent/'recon.json'
            with open(recon_json, 'w') as rj:
                json.dump(recon, rj)
        if trans:
            trans_json = self.path/'transforms.json'
            if append:
                with open(trans_json) as tj:
                    existing = json.load(tj)
                if not isinstance(existing, list):
                    # placeholder comment written at create time
                    existing = []
                if isinstance(trans, dict):
                    trans = existing + [trans]
                else:
                    trans = existing + list(trans)
            with open(trans_json, 'w') as tj:
                json.dump(trans, tj)


    # ------------------------------------------------------------------ load

    def _load_transform(self, entry, params, stack, channel):
        t_type, t_format, t_name = entry['type'], entry['format'], entry['name']

        if 'affine' == t_type and 'tfm' == t_format:
            stack_idx, channel_idx = self._resolve_indices(
                params, stack, channel, both_required=True,
                message='Loading affine transform requires [stack_index, channel_index] in params.')
            trans_path = self.path/t_name/f'{stack_idx}'/f'{channel_idx}'/f'{t_type}.{t_format}'
            if not trans_path.exists():
                raise FileNotFoundError(f'The transform file {trans_path} does not exist.')
            return sitk.ReadTransform(trans_path)

        if 'dense displacement field' == t_type and 'mha' == t_format:
            channel_idx = self._resolve_indices(
                params, stack, channel, both_required=False,
                message="Loading ddf transform takes [] (slice-level) or [channel_index] in params.")
            if channel_idx is None:
                trans_path = self.path/t_name/f'{t_type}.{t_format}'
            else:
                trans_path = self.path/t_name/f'{channel_idx}'/f'{t_type}.{t_format}'
            if not trans_path.exists():
                raise FileNotFoundError(f'The transform file {trans_path} does not exist.')
            field = sitk.ReadImage(trans_path)
            return sitk.DisplacementFieldTransform(
                sitk.Cast(field, sitk.sitkVectorFloat64))

        raise NotImplementedError(
            f"Transform type '{t_type}' with format '{t_format}' is not supported yet; "
            f"supported: affine+tfm, 'dense displacement field'+mha.")


    def _invert(self, transform, entry):

        if isinstance(transform, sitk.DisplacementFieldTransform):
            raise RuntimeError(
                'Displacement-field transforms have no analytic inverse; '
                'store the required direction explicitly or invert numerically '
                '(see visor-resample).')
        try:
            return transform.GetInverse()
        except RuntimeError as e:
            raise RuntimeError(
                f"Cannot invert transform {entry['name']}: {e}") from e


    # ----------------------------------------------------------------- utils

    @staticmethod
    def _resolve_indices(params, stack, channel, both_required, message):

        indices = []
        if params is not None:
            indices = list(params)
        elif stack is not None:
            indices = [stack]
            if channel is not None:
                indices.append(channel)
        elif channel is not None:
            indices = [channel]

        if both_required:
            if len(indices) != 2:
                raise ValueError(message)
            return indices[0], indices[1]
        else:
            if len(indices) > 1:
                raise ValueError(message)
            return indices[0] if indices else None


    @staticmethod
    def _affine_params(params, transform, stack, channel):
        """Return (stack, channel, matrix(9), translation(3)) from the legacy
        params list or a transform object + indices."""

        if transform is not None:
            if stack is None or channel is None:
                raise ValueError(
                    'Saving an affine from a transform object requires stack and channel.')
            t = sitk.AffineTransform(3)
            t.SetParameters(sitk.Transform.GetParameters(transform))
            return stack, channel, list(t.GetMatrix()), list(t.GetTranslation())
        if params is None:
            raise ValueError(
                'Saving affine transform requires [stack_index, channel_index, affine_mat, affine_vec] in params.')
        params = list(params)
        if 14 != len(params):
            raise ValueError(
                'Saving affine transform requires [stack_index, channel_index, affine_mat, affine_vec] in params.')
        return params[0], params[1], params[2:11], params[11:14]


    @staticmethod
    def _ddf_field(transform):

        if isinstance(transform, sitk.DisplacementFieldTransform):
            return transform.GetDisplacementField()
        if isinstance(transform, sitk.Image):
            if transform.GetPixelID() not in (sitk.sitkVectorFloat32, sitk.sitkVectorFloat64):
                raise ValueError('ddf transform image must have a vector float pixel type.')
            return transform
        raise ValueError(
            "Saving a ddf requires transform= as a SimpleITK vector image or "
            "DisplacementFieldTransform.")
