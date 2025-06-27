from pathlib import Path
import json
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
import numpy

class Transform:

    def __init__(self, vsr_path:str|Path, recon_version:str,
                 slice_name:str, transform_name:str):
        """
        Constructor of Transform

        Parameters:
            vsr_path:       path to the .vsr file
            recon_version:  reconstruction version, see visor.info()['recon_versions']
            slice_name:     name of slice, see visor.list_transform()
            transform_name: name of transform, see visor.list_transform()
        """
        slice_path = Path(vsr_path)/'visor_recon_transforms'/recon_version/slice_name
        transform_path = Path(vsr_path)/'visor_recon_transforms'/recon_version/slice_name/transform_name
        transform_path.mkdir(parents=True, exist_ok=True)

        self.path = transform_path

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
