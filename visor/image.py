from pathlib import Path
import json
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
from zarr.codecs import BloscCodec
import numpy

class Image:

    def __init__(
            self, path:str|Path, type:str,
            name:str, mode:str='r'):
        """
        Constructor of Image

        Parameters:
            path: path to the .vsr file
            type: image type, see visor.info()['image_types']
            name: name of slice, see visor.list_image()
            mode: mode pass to zarr.open()
        """

        image_path = Path(path)/f'visor_{type}_images'/f'{name}.zarr'

        if 'r' == mode:
            if not image_path.exists() or not image_path.is_dir():
                raise NotADirectoryError(f'The image path {image_path} is not valid.')
        elif 'w-' == mode:
            if image_path.exists():
                raise FileExistsError(f'The image path {image_path} already exist.')
        else:
            raise ValueError(f'Invalid mode \'{mode}\': mode values could be \'r\' or \'w-\'.')

        self.path   = image_path
        self.mode   = mode
        self.zgroup = zarr.open_group(image_path, mode=mode)
        self.attrs  = self.zgroup.attrs.asdict()

    def label_to_index(self, filter:str, label:str):
        """
        Get index from label of filter stack/channel

        Parameters:
            filter: stack or channel
            label:  label of named stack or channel

        Returns:
            int
        """
        v_meta = self.attrs['visor']
        if 'stack' == filter:
            for s in v_meta['visor_stacks']:
                if s['label'] == label:
                    return s['index']
            raise ValueError(f'The visor_stack {label} does not exist.')
        elif 'channel' == filter:
            for s in v_meta['channels']:
                if s['wavelength'] == label:
                    return s['index']
            raise ValueError(f'The channel {label} does not exist.')
        else:
            raise ValueError(f'Invalid filter {filter}. Must be stack or channel')

    def open(self, resolution:str):
        """
        Open a zarr array by resolution

        Parameters:
            resolution: resolution level, see visor.list_image()

        Returns:
            zarr.Array
        """

        return self.zgroup[str(resolution)]
    
    def create(
            self, resolution:str, attrs:dict=None, 
            dtype:str=None, shape:tuple=None,
            shard_size:tuple=None, chunk_size:tuple=None,
            compressors:BloscCodec=None):

        array_path = self.path/str(resolution)

        if array_path.is_dir():
            raise FileExistsError(f'The array {array_path} already exist.')
        zarr.create_array(
            store=str(self.path),
            name=str(resolution),
            dtype=dtype,
            shape=shape,
            shards=shard_size,
            chunks=chunk_size,
            compressors=compressors,
        )
        self.zgroup.attrs.update(attrs or {})

        return self.zgroup[str(resolution)]
