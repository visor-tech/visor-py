from pathlib import Path
import json
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
from zarr.codecs import BloscCodec

class Image:

    def __init__(
            self, path:str|Path, type:str, name:str, mode:str='r',
            attr:dict=None, resolution:str=None, dtype:str=None,
            shape:tuple=None, shard_size:tuple=None, chunk_size:tuple=None,
            compressors:BloscCodec=None):
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
            self.store = zarr.open(image_path, mode='r')
        elif 'w-' == mode: 
            if image_path.exists():
                raise FileExistsError(f'The image path {image_path} already exist.')
            zarr.create_array(
                store=str(image_path),
                name=resolution,
                dtype=dtype,
                shape=shape,
                shards=shard_size,
                chunks=chunk_size,
                compressors=compressors,
            )
            with open(image_path/'zarr.json', 'r+', encoding='utf-8') as zj:
                zmeta = json.load(zj)
                zmeta['attributes'] = attr
                json.dump(zmeta, zj)
            self.store = zarr.open(image_path, mode='w')
        else:
            raise ValueError(f'Invalid mode \'{mode}\': mode values could be \'r\' or \'w-\'.')

        self.path = image_path
        self.mode = mode

    def load(self, resolution:str, stack:str=None, channel:str=None):
        """
        Load Image, as a zarr array

        Parameters:
            resolution: resolution level, see visor.list_image()
            stack:      visor stack label
            channel:    channel wavelength

        Returns:
            zarr.Array
        """

        image_array = self.store[resolution]

        return image_array
    
    def save(self, resolution:str, stack:str=None, channel:str=None):
        pass
