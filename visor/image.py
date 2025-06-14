from pathlib import Path
import json
import numpy as np
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

# Schema Reference: https://visor-tech.github.io/visor-data-schema

class Image:

    def __init__(self, path:Path):
        """
        Constructor of Image

        Parameters:
            path : the .zarr image file path
        """
        if path.suffix != '.zarr':
            raise ValueError(f'The path {path} does not have .zarr extension.')
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f'The path {path} is not a directory.')
        self.path = path
        meta_file = path/'zarr.json'

    def load_resolution(self, resolution='0'):
        return Array
    
    def write(arr=slice_arr,
              img_type='raw',
              file=f'slice_{str(idx_slice)}_10x',
              resolution=0,
              img_info=img_info,
              arr_info=arr_info,
              chunk_size=CHUNK_SIZE,
              shard_size=SHARD_SIZE,
              selected=selected)
 ##   
    array = img.resolution('0')
    array:zarr.array = img.load_resolution('0')
    array[0,0,:1,:1,:1]

    dask_arr = da.from_array(img.load_resolution('0'))
    dask_arr = da.from_zarr(img.path,prefix='0/c')


    visor.to_vsr(path, image)
    to_zarr(arry, zarr_file_path)
#-----

    with open('xxx.json', 'r') as jf:
        json.dump(jf, {})

#-----

    # vsr = visor.create_vsr(path)
    vsr = visor.open_vsr(path, mode)
    vsr.info()
    vsr.image_info()

    for raw_image_info in vsr.image_info(type='raw'):

      raw_image_name = raw_image_info['name']
      raw_image = vsr.image(
          'raw', 
          raw_image_name, 
      )

      personnel = 'xxx'
      date = date_now()
      compr_image_name = f'{personnel}_{raw_image_name.split('.zarr')[0]}_{date}'
      compress_image = vsr.image(
          'compr',
          compr_image_name,
          compressor='ffmpeg',
          shape=(2,1,3,3,3)
      )
      
      for res in raw_image.info()['resolutions'].keys():
        for st in raw_image.info()['stacks'].keys():
          for ch in raw_image.info()['channels'].keys():
              arr = raw_image.read(resolution=res,stack=st,channel=ch)
              compress_image.write(arr,resolution=res,stack=st,channel=ch)
        # vsr.add_image(compress_image)

#------------
    vsr = visor.from_vsr(path)

    for raw_image_info in vsr.image_info(type='raw'):

      raw_image_name = raw_image_info['name']
      img = vsr.load_image(type='raw', name=raw_image_name)

      personnel = 'xxx'
      date = date_now()
      compr_image_name = f'{personnel}_{raw_image_name.split('.zarr')[0]}_{date}'
      for res in img.info()['resolutions'].keys():
        arr = img.load_resolution(res)
        vsr.dump_image(
            arr, 
            type='compr', 
            name=compr_image_name,
            compressor='ffmpeg',
            resolution=res,
            img_info=img_info,
            arr_info=arr_info,
            chunk_size=CHUNK_SIZE,
            shard_size=SHARD_SIZE
        )



        # vsr.add_image(compress_image)

    # vsr.open_image(name, type)
    # vsr.add_image(image):
    #   path = f'visor_{type}_images'/name
    #   mkdir(path)
    #   zarr.write(compression=ffmpeg)
    

    # image.update()

##
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


    @staticmethod
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
        return [c['wavelength'] for c in meta['attributes']['visor']['channels']]

    @staticmethod
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
        return [s['label'] for s in meta['attributes']['visor']['visor_stacks']]


    @staticmethod
    def _get_resolutions(meta_file):
        """
        Private method to get resolution list from metadata file

        Parameters:
            meta_file : the metadata file path

        Returns:
            List of resolution objects
        """

        with open(meta_file) as mf:
            meta = json.load(mf)
        return meta['attributes']['ome']['multiscales'][0]['datasets']

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


    def write(self, arr:np.ndarray, img_type:str, file:str,
                    resolution:int, img_info:dict, arr_info:dict,
                    chunk_size:tuple, shard_size:tuple,
                    selected:dict=None, overwrite:bool=False,
                    compressor:zarr.codecs.BloscCodec=zarr.codecs.BloscCodec(cname='zstd', clevel=5)):
        """
        Read Array from Image

        Parameters:
            arr         : numpy array to write
            img_type    : visor image type, e.g. 'raw', 'projn', etc.
            file        : file name without extension
            resolution  : resolution level
            img_info    : metadata to write to info.json
            arr_info    : metadata to write to zarr.json
            selected    : optional metadata to write to selected.json
            compressor  : compression algorithm to use for zarr,
                          default to Blosc(cname='zstd', clevel=5).
            overwrite   : overwrite the existing file if True, default to False.
        """

        if self.mode != 'w':
            raise PermissionError('"write" method is only available in "w" mode.')
        
        zarr_file = self.path/f'visor_{img_type}_images'/f'{file}.zarr'

        zarr_arr = zarr.create_array(
            store=str(zarr_file),
            name=str(resolution),
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=chunk_size,
            shards=shard_size,
            compressors=compressor,
            overwrite=overwrite
        )
        zarr_arr[...] = arr
        
        img_info_file = self.path/'info.json'
        with open(img_info_file, 'w') as iif:
            json.dump(img_info, iif, indent=2)

        arr_info_file = zarr_file/'zarr.json'
        zarr_json = {}
        with open(arr_info_file, 'r') as aif:
            zarr_json = json.load(aif)
        with open(arr_info_file, 'w') as aif:
            zarr_json['attributes'] = arr_info
            json.dump(zarr_json, aif, indent=2)

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

        # zarr.json
        zarr_json_file = path/str(resolution)/'zarr.json'
        if not zarr_json_file.exists():
            raise FileNotFoundError(f'Must contain a zarr.json file in {path/str(resolution)} directory.')
        
        # info
        meta_file = path/'zarr.json'
        with open(meta_file) as mf:
            self.info = json.load(mf)

        # dictionares of named dimension
        self.channel_map = {}
        self.stack_map = {}
        if 'visor' in self.info['attributes']:
            if 'channels' in self.info['attributes']['visor']:
                for channel in self.info['attributes']['visor']['channels']:
                    self.channel_map[channel['wavelength']] = channel['index']
            if 'visor_stacks' in self.info['attributes']['visor']:
                for stack in self.info['attributes']['visor']['visor_stacks']:
                    self.stack_map[stack['label']] = stack['index']

        # array
        _store = zarr.storage.LocalStore(path/str(resolution))
        self.array = zarr.open(_store, mode='r')


    def read(self, channel=None, stack=None):        
        """
        Read a subarray from this array with optional `vs` and `ch` dimensions.
        
        Parameters:
            stack: str or None, the visor_stack label
                Get index from stack_map. If None, take all `vs` subarrays.
            channel: str or None, the channel wavelength
                Get index from channel_map. If None, take all `ch` subarrays.
        
        Returns:
            numpy.ndarray
        """
        # Total dimensions in the array
        ndim = self.array.ndim

        # Define slices for each dimension
        arr_slices = [slice(None)] * ndim

        # Validate `stack` and `channel` based on array dimensions
        if ndim == 5:  # (vs, ch, z, y, x)
            if (stack is not None) and (stack not in self.stack_map):
                raise KeyError(f'stack {stack} does not exist, valid stacks: {list(self.stack_map.keys())}')
            if (channel is not None) and (channel not in self.channel_map):
                raise KeyError(f'channel {channel} does not exist, valid channels: {list(self.channel_map.keys())}')
        elif ndim == 4:  # (ch, z, y, x)
            if (channel is not None) and (channel not in self.channel_map):
                raise KeyError(f'channel {channel} does not exist, valid channels: {list(self.channel_map.keys())}')
        elif ndim == 3:  # (z, y, x)
            if stack is not None or channel is not None:
                raise IndexError("'vs' or 'ch' dimensions do not exist in a 3D array")

        # Map the dimensions dynamically based on `ndim`
        if ndim == 5:  # (vs, ch, z, y, x)
            if stack is not None:
                idx = self.stack_map[stack]
                arr_slices[0] = slice(idx, idx+1)
            if channel is not None:
                idx = self.channel_map[channel]
                arr_slices[1] = slice(idx, idx+1)
        elif ndim == 4:  # (ch, z, y, x)
            if channel is not None:
                idx = self.channel_map[channel]
                arr_slices[0] = slice(idx, idx+1)
        elif ndim == 3:  # (z, y, x)
            pass  # No `vs` or `ch` dimensions to handle

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
