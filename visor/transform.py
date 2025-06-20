from pathlib import Path
import json
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
import numpy

class Transform:

    def __init__(self, path:Path):
        """
        Constructor of Transform

        Parameters:
            path:    path to the .vsr file
            version: reconstruction version, see visor.info()['recon_versions']
            name:    name of slice, see visor.list_image()

        """
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f'Version {path} does not exist.')
        self.path = path

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


def transform(path:str|Path, version:str):
    """
    Create an Transform

    Parameters:
        path:    path to the .vsr file
        version: reconstruction version, see visor.info()['recon_versions']

    Returns:
        Transform
    """
    transform_path = Path(path)/'visor_recon_transforms'/version
    return Transform(transform_path)
