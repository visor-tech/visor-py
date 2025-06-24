from pathlib import Path
import json
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

class _VSR:

    def __init__(self, vsr_path:Path):
        """
        Constructor of VSR

        Parameters:
            vsr_path : path to the .vsr file
        """
        if vsr_path.suffix != '.vsr':
            raise ValueError(f'The path {vsr_path} does not have .vsr extension.')
        if not vsr_path.exists() or not vsr_path.is_dir():
            raise NotADirectoryError(f'The path {vsr_path} is not a directory.')
        self.path = vsr_path
        self.image_types = [d.name.split('_')[1] for d in self.path.glob('visor_*_images')]
        self.recon_versions = []
        if (self.path/'visor_recon_transforms').is_dir():
            self.recon_versions = \
                [i.name for i in (self.path/'visor_recon_transforms').iterdir() if i.is_dir()]

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
            meta_file : path to the metadata file

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
            meta_file : path to the metadata file

        Returns:
            List of resolutions
        """

        resolutions = {}

        with open(meta_file) as mf:
            meta = json.load(mf)
        for r in meta['attributes']['ome']['multiscales'][0]['datasets']:
            resolutions[r['path']] = r['coordinateTransformations'][0]['scale']

        return resolutions

    def transforms(self):
        """
        Collect transforms in .vsr file

        Returns:
            Collection of transform descriptions
        """
        transforms = {}

        dir = self.path/f'visor_recon_transforms'
        for version in self.recon_versions:
            with open(dir/version/'recon.json') as rf:
                recon_info = json.load(rf)
            transforms[version] = {
                "spaces": recon_info['spaces'],
                "slices": recon_info['slices']
            }

        return transforms


def info(vsr_path:str|Path):
    """
    Get information of the VSR file

    Parameters:
        path: path to the .vsr file

    Returns:
        JSON like object
    """
    vsr = _VSR(Path(vsr_path))

    info_file = vsr.path/'info.json'
    if not info_file.exists():
        raise FileNotFoundError(f'Metadata file info.json is not found in {path}.')
    with open(info_file) as f:
        info = json.load(f)

    info['image_types'] = vsr.image_types
    info['recon_versions'] = vsr.recon_versions

    return info


def list_image(vsr_path:str|Path, image_type=None):
    """
    List images, optionally by filters

    Parameters:
        vsr_path:   path to the .vsr file
        image_type: image type, see visor.info()['image_types']

    Returns:
        Collection of image descriptions
    """
    vsr = _VSR(Path(vsr_path))
    images = vsr.images()

    if image_type:
        images = images[image_type]

    return images


def list_transform(vsr_path:str|Path, recon_version=None):
    """
    List transforms, optionally by filters

    Parameters:
        vsr_path:      path to the .vsr file
        recon_version: reconstruction version, see visor.info()['recon_versions']

    Returns:
        Collection of transform descriptions
    """
    vsr = _VSR(Path(vsr_path))
    transforms = vsr.transforms()

    if recon_version:
        transforms = transforms[recon_version]

    return transforms    


def create_vsr(vsr_path:str|Path):
    """
    Create a .vsr directory

    Parameters:
        vsr_path : path to the .vsr file
    """
    # Validate vsr path
    if vsr_path.suffix != '.vsr':
        raise ValueError(f'The path {vsr_path} is not valid, must contain .vsr extension.')
    
    # Create vsr directory
    try:
        vsr_path.mkdir()
    except FileExistsError:
        raise FileExistsError(f'VSR {vsr_path} already exists.')

    # Create an empty info.json file with comment
    with open(vsr_path/'info.json', 'w') as info_json:
        info_json.write('{\n  "_comment": "see https://visor-tech.github.io/visor-data-schema/"\n}')
    
    # Create visor_raw_images directory
    raw_image_path = vsr_path/'visor_raw_images'
    raw_image_path.mkdir()
    with open(raw_image_path/'selected.json', 'w') as selected_json:
        selected_json.write('{\n  "_comment": "see https://visor-tech.github.io/visor-data-schema/"\n}')
