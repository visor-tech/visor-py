from pathlib import Path
import json
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
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
        vsr_path = Path(vsr_path)
        # Validate vsr path
        if vsr_path.suffix != '.vsr':
            raise ValueError(f'The path {vsr_path} is not valid, must contain .vsr extension.')
        if not vsr_path.exists() or not vsr_path.is_dir():
            raise NotADirectoryError(f'The path {vsr_path} is not a directory.')
        
        transform_path = vsr_path/'visor_recon_transforms'/recon_version/slice_name
        if create:
            transform_path.mkdir(parents=True, exist_ok=True)
            with open(transform_path/'transforms.json', 'w') as trans_json:
                trans_json.write('{\n  "_comment": "see https://visor-tech.github.io/visor-data-schema/"\n}')
        if not transform_path.exists() or not transform_path.is_dir():
            raise NotADirectoryError(f'The path {transform_path} is not a directory.')
        self.path = transform_path


    def load(self, from_space:str, to_space:str):
        """
        Load Transform

        Parameters:
            from_space: source space name
            to_space:   target space name

        Return:
            depends on transform type and format
        """
        t_meta_file = self.path/'transforms.json'
        if not t_meta_file.exists():
            raise FileNotFoundError(f'Metadata file transforms.json is not found in {self.path}.')
        with open(t_meta_file) as f:
            t_list = json.load(f)

        t_name = f'{from_space}_to_{to_space}'
        t_inv_name = f'{to_space}_to_{from_space}'
        for t in t_list:
            if t_name == t['name']:
                return self._load_trans(
                    t_name=t['name'],
                    t_type=t['type'],
                    t_format=t['format'],
                )
            elif t_inv_name == t['name']:
                return self._load_inv_trans(
                    t_name=t['name'],
                    t_type=t['type'],
                    t_format=t['format'],
                )
            else:
                raise FileNotFoundError(f'Transform {from_space}_to_{to_space} is not in {self.path}.')

    def _load_trans(self, t_name:str, t_type:str, t_format:str):
        trans_path = self.path/t_name/f'{t_type}.{t_format}'
        if not trans_path.exists():
            raise NotADirectoryError(f'The path {trans_path} is not a directory.')
        
        if 'affine' == t_type and 'tfm' == t_format:
            return sitk.ReadTransform(trans_path)

    def _load_inv_trans(self, t_name:str, t_type:str, t_format:str):
        pass

    def save(self):
        pass

    def update_meta(self, recon:dict=None, trans:dict=None):
        """
        Update transforms.json

        Parameters:
            attrs: new attributes
        """  
        if recon:
            recon_json = self.path.parent/'recon.json'
            with open(recon_json, 'w') as rj:
                json.dump(recon, rj)
        if trans:
            trans_json = self.path/'transforms.json'
            with open(trans_json, 'w') as tj:
                json.dump(trans, tj)


    def apply(self, roi:tuple[slice,...], from_space:str, to_space:str):
        """
        Apply Transform

        Parameters:
            roi:        tuple of slices, represents ROI coordinates in source space
            from_space: name of the source space
            to_space:   name of the target space

        Returns:
            tuple of slices

        Note:
            single point is a special case of ROI
        """

        pass
