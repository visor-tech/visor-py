from pathlib import Path
import os
import json
import zarr
import dask.array as da
from numcodecs import Blosc

# Schema Reference: https://visor-tech.github.io/visor-data-schema

class Image:

    def __init__(self, path:Path, mode:str):
        """
        Constructor of Image

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

        # info
        info_file = path/'info.json'
        if not info_file.exists():
            if 'w' == mode:
                info_file.touch()
                info_file.write_text('{}')
            else:
                raise FileNotFoundError(f'Metadata file info.json is not found in {path}.')
        with open(info_file) as f:
            self.info = json.load(f)

        # file structure
        self.image_types = self._get_image_types()
        self.transforms = self._get_transforms()
        self.image_files = {}
        for image_type in self.image_types:
            dir = path/f'visor_{image_type}_images'
            if 'raw' == image_type:
                with open(dir/'selected.json') as sf:
                    self.image_files[image_type] = json.load(sf)
            else:
                self.image_files[image_type] = [{
                    "path": d,
                    "channels": self._get_channels(dir/f/'.zattr')
                } for d in dir.iterdir() if d.suffix == '.zarr']


    def _get_image_types(self):
        """
        Private method to get image type list

        Returns:
            List of image types
        """

        return [d.name.split('_')[1] for d in self.path.glob('visor_*_images')]


    def _get_transforms(self):
        """
        Private method to get transform list

        Returns:
            List of transforms
        """

        transforms = []

        transforms_dir = self.path/'visor_recon_transforms'
        if transforms_dir.is_dir():
            transforms = transforms_dir.iterdir()

        return transforms


    def _get_channels(meta_file):
        """
        Private method to get channel list from metadata file

        Parameters:
            meta_file : the metadata file path

        Returns:
            List of channel wavelengths
        """

        with open(meta_file) as mf:
            meta = json.load(mf)
        return [c['wavelength'] for c in meta['channels']]


    def _get_stacks(meta_file):
        """
        Private method to get stack list

        Parameters:
            meta_file : the metadata file path
        
        Returns:
            List of stack labels
        """

        with open(meta_file) as mf:
            meta = json.load(mf)
        return [s['label'] for s in meta['visor_stacks']]


    def list(self, img_type=None):
        """
        List image files

        Parameters:
            img_type : optional filter of image type

        Returns:
            Dictionary of image files
        """

        if None == img_type:
            return self.image_files
        elif img_type not in self.image_files:
            return {}
        else:
            return self.image_files[img_type]


    def read(self, img_type:str, zarr_file:str, resolution:int):
        """
        Read Array from Image

        Parameters:
            img_type   : visor image type, e.g. 'raw', 'projn', etc.
            zarr_file  : zarr file name with '.zarr' extension
            resolution : resolution level

        Returns:
            Array
        """

        return Array(path=self.path/f'visor_{img_type}_images'/zarr_file,
                     resolution=resolution)


    def write(self, arr:da, img_type:str, file:str,
                    resolution:int, img_info:dict,
                    arr_info:dict, selected:dict=None,
                    compression:str='zstd', clevel:int=5):
        """
        Read Array from Image

        Parameters:
            arr         : dask array to write
            img_type    : visor image type, e.g. 'raw', 'projn', etc.
            file        : file name without extension
            resolution  : resolution level
            img_info    : metadata to write to info.json
            arr_info    : metadata to write to .zattrs
            selected    : optional metadata to write to selected.json
            compression : compression algorithm to use for zarr, default to 'zstd'
            clevel      : compression level to use for zarr, default to 5
        """

        if self.mode != 'w':
            raise PermissionError('"write" method is only available in "w" mode.')
        
        zarr_file = self.path/f'visor_{img_type}_images'/f'{file}.zarr'
        component = str(resolution)
        da.to_zarr(arr, zarr_file,
                   dimension_separator=os.sep,
                   component=component,
                   compressor=Blosc(cname=compression, clevel=clevel))
        
        img_info_file = self.path/'info.json'
        with open(img_info_file, 'w') as iif:
            json.dump(img_info, iif, indent=2)

        arr_info_file = zarr_file/'.zattrs'
        with open(arr_info_file, 'w') as aif:
            json.dump(arr_info, aif, indent=2)

        if selected and 'raw' == img_type:
            selected_file = self.path/'visor_raw_images'/'selected.json'
            with open(selected_file, 'w') as sf:
                json.dump(selected, sf, indent=2)


class Array:

    def __init__(self, path:Path, resolution:int):
        """
        Constructor of Array

        Parameters:
            path : the .zarr file path
            resolution : resolution level
        """
        # path
        if path.suffix != '.zarr':
            raise ValueError(f'{path} is not a zarr file.')
        if not path.is_dir():
            raise NotADirectoryError(f'The path {path} is not a directory.')
        self.path = path
        
        # resolution
        self.resolution = resolution

        # .zarray
        zarray_file = path/str(resolution)/'.zarray'
        if not zarray_file.exists():
            raise FileNotFoundError(f'Must contain a .zarray file in {path} directory.')
        
        # info
        meta_file = path/'.zattrs'
        with open(meta_file) as mf:
            self.info = json.load(mf)

        # dictionares of named dimension
        self.channel_map = {}
        if 'channels' in self.info:
            for channel in self.info['channels']:
                self.channel_map[channel['wavelength']] = channel['index']
        self.stack_map = {}
        if 'visor_stacks' in self.info:
            for stack in self.info['visor_stacks']:
                self.stack_map[stack['label']] = stack['index']

        # array
        _store = zarr.DirectoryStore(path/str(resolution))
        z_array = zarr.open(_store, mode='r')
        self.array = da.from_array(z_array, chunks=z_array.chunks)


    def read(self, channel=None, stack=None):        
        """
        Read a subarray from this array with optional `s` and `c` dimensions.
        
        Parameters:
            stack: str or None, the visor_stack label
                Get index from stack_map. If None, take all `s` subarrays.
            channel: str or None, the channel wavelength
                Get index from channel_map. If None, take all `c` subarrays.
        
        Returns:
            dask.array
        """
        # Total dimensions in the array
        ndim = self.array.ndim

        # Define slices for each dimension
        arr_slices = [slice(None)] * ndim

        # Validate `stack` and `channel` based on array dimensions
        if ndim == 5:  # (s, c, z, y, x)
            if (stack is not None) and (stack not in self.stack_map):
                raise KeyError(f'stack {stack} does not exist, valid stacks: {list(self.stack_map.keys())}')
            if (channel is not None) and (channel not in self.channel_map):
                raise KeyError(f'channel {channel} does not exist, valid channels: {list(self.channel_map.keys())}')
        elif ndim == 4:  # (c, z, y, x)
            if (channel is not None) and (channel not in self.channel_map):
                raise KeyError(f'channel {channel} does not exist, valid channels: {list(self.channel_map.keys())}')
        elif ndim == 3:  # (z, y, x)
            if stack is not None or channel is not None:
                raise IndexError("'s' or 'c' dimensions do not exist in a 3D array")

        # Map the dimensions dynamically based on `ndim`
        if ndim == 5:  # (s, c, z, y, x)
            if stack is not None:
                arr_slices[0] = self.stack_map[stack]
            if channel is not None:
                arr_slices[1] = self.channel_map[channel]
        elif ndim == 4:  # (c, z, y, x)
            if channel is not None:
                arr_slices[0] = self.channel_map[channel]
        elif ndim == 3:  # (z, y, x)
            pass  # No `s` or `c` dimensions to handle

        # Apply the arr_slices to the array
        return self.array[tuple(arr_slices)]


def open_vsr(path:str, mode:str):
    """
    Open vsr file, as an Image object

    Parameters:
        path : the .vsr file path
        mode : 'r' for read_only
               'w' for write

    Returns:
        Image
    """
    return Image(Path(path), mode)
