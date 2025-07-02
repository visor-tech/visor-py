from pathlib import Path
import json
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
import numpy

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
        if not transform_path.exists() or not transform_path.is_dir():
            raise NotADirectoryError(f'The path {transform_path} is not a directory.')

        self.path = transform_path


    def load(self):
        """

        """
        pass
        # switch self.type
        # case 'tfm':
        #     stik.Tranform.read()
        # case 'zarr':
            

    def save(self):
        pass

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
