from pathlib import Path
import json

class VSR:

    def __init__(self, path:Path, mode:str):
        """
        Constructor of Vsr

        Parameters:
            path : the .vsr file path
            mode : 'r' for read_only
                   'w' for write
        """
        # path
        path = Path(path)
        if path.suffix != '.vsr':
            raise ValueError(f'The path {path} does not have .vsr extension.')
        if not path.exists() or not path.is_dir():
            if 'w' == mode:
                path.mkdir(parents=True, exist_ok=True)
            else: 
                raise NotADirectoryError(f'The path {path} is not a directory.')
        self.path = path
        self.mode = mode


    def _get_transform_versions(self):
        """
        Private method to get transform version list

        Returns:
            List of transform versions
        """

        transform_versions = []

        transforms_dir = self.path/'visor_recon_transforms'
        if transforms_dir.is_dir():
            transform_versions = transforms_dir.iterdir()

        return transform_versions


    def _get_image_files(self):
        """
        Private method to get image file list

        Returns:
            List of image files
        """

        image_files = []

        image_types = [d.name.split('_')[1] for d in self.path.glob('visor_*_images')]
        for image_type in image_types:
            dir = self.path/f'visor_{image_type}_images'
            if 'raw' == image_type:
                with open(dir/'selected.json') as sf:
                    self.image_files[image_type] = json.load(sf)
                    for idx in range(len(self.image_files[image_type])):
                        d = self.image_files[image_type][idx]["path"]
                        self.image_files[image_type][idx]["resolutions"] = \
                            self._get_resolutions(dir/d/'zarr.json')
            else:
                self.image_files[image_type] = [{
                    "path": d.name,
                    "channels": self._get_channels(dir/d/'zarr.json'),
                    "resolutions": self._get_resolutions(dir/d/'zarr.json')
                } for d in dir.iterdir() if d.suffix == '.zarr']

        return image_files


    def get_info(self):
        """
        Get vsr file information

        Returns:
            JSON like object
        """
        # metadata
        info_file = self.path/'info.json'
        if not info_file.exists():
            raise FileNotFoundError(f'Metadata file info.json is not found in {self.path}.')
        with open(info_file) as f:
            info = json.load(f)

        # file structure
        info.transform_versions = self._get_transform_versions()
        info.image_files = self._get_image_files()

        return info



def open(path:str, mode:str='r'):
    """
    Open vsr file, as an Vsr object

    Parameters:
        path : the .vsr file path
        mode : 'r' for read_only
               'w' for write

    Returns:
        Vsr
    """
    return VSR(Path(path), mode)
