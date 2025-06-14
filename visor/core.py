from pathlib import Path
import json

class _VSR:

    def __init__(self, path:Path):
        """
        Constructor of VSR

        Parameters:
            path : the .vsr file path
        """
        if path.suffix != '.vsr':
            raise ValueError(f'The path {path} does not have .vsr extension.')
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f'The path {path} is not a directory.')
        self.path = path
        self.image_types = [d.name.split('_')[1] for d in self.path.glob('visor_*_images')]
        self.transform_versions = []
        if (self.path/'visor_recon_transforms').is_dir():
            self.transform_versions = \
                [i.name for i in (path/'visor_recon_transforms').iterdir() if i.is_dir()]

    def images(self):
        """
        Collect images in .vsr file

        Returns:
            Collection of image descriptions
        """
        images = {}
        for image_type in self.image_types:

            dir = self.path/f'visor_{image_type}_images'

            if 'raw' == image_type:
                with open(dir/'selected.json') as sf:
                    images['raw'] = json.load(sf)
                for i in images['raw']:
                    i['resolutions'] = self._resolutions(dir/f'{i['name']}.zarr'/'zarr.json')
            else:
                images[image_type] = [{
                    'name': d.name.replace('.zarr',''),
                    'channels': self._channels(dir/d/'zarr.json'),
                    'resolutions': self._resolutions(dir/d/'zarr.json')
                } for d in dir.iterdir() if d.suffix == '.zarr']

        return images

    @staticmethod
    def _channels(meta_file):
        """
        Private method to get channel list from metadata file

        Parameters:
            meta_file : the metadata file path

        Returns:
            List of channel wavelengths
        """

        with open(meta_file) as mf:
            meta = json.load(mf)
        return [c['wavelength'] for c in meta['attributes']['visor']['channels']]

    @staticmethod
    def _resolutions(meta_file):
        """
        Private method to get resolutions list from metadata file

        Parameters:
            meta_file : the metadata file path

        Returns:
            List of resolutions
        """

        resolutions = {}

        with open(meta_file) as mf:
            meta = json.load(mf)
        for r in meta['attributes']['ome']['multiscales'][0]['datasets']:
            resolutions[r['path']] = r['coordinateTransformations'][0]['scale']

        return resolutions


def info(path:str|Path):
    """
    Get information of the VSR file

    Returns:
        JSON like object
    """
    vsr = _VSR(Path(path))

    info_file = vsr.path/'info.json'
    if not info_file.exists():
        raise FileNotFoundError(f'Metadata file info.json is not found in {path}.')
    with open(info_file) as f:
        info = json.load(f)

    info['image_types'] = vsr.image_types
    info['transform_versions'] = vsr.transform_versions

    return info

def list_image(path:str|Path, type=None, channel=None):
    """
    List image, optionally by filters

    Parameters:
        type: the image type, see visor.info()['image_types']
        channel: the channel wavelength

    Returns:
        Collection of image descriptions
    """
    vsr = _VSR(Path(path))
    images = vsr.images()

    if channel:
        for image_type in images:
            images[image_type] = [fo for fo in images[image_type] if channel in fo['channels']]

    if type:
        images = images[type]

    return images

def list_transform():
    pass

def create(path:str):
    """
    Create vsr file, return a VSR object

    Parameters:
        path : the .vsr file path

    Returns:
        VSR object
    """

    vsr_path = Path(path)

    # Validate vsr path
    if vsr_path.suffix != '.vsr':
        raise ValueError(f'The path {path} is not valid, must contain .vsr extension.')
    
    # Create vsr directory
    try:
        vsr_path.mkdir()
    except FileExistsError:
        raise FileExistsError(f'VSR {path} already exists.')

    # Create an empty info.json file with comment
    with open(vsr_path/'info.json', 'w') as info_json:
        info_json.write('{\n  "_comment": "see https://visor-tech.github.io/visor-data-schema/"\n}')
    
    # Create visor_raw_images directory
    raw_image_path = vsr_path/'visor_raw_images'
    raw_image_path.mkdir()
    with open(raw_image_path/'selected.json', 'w') as selected_json:
        selected_json.write('{\n  "_comment": "see https://visor-tech.github.io/visor-data-schema/"\n}')

    return VSR(vsr_path)