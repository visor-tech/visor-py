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
        with open(image_path/'zarr.json', 'r') as zj:
            self.attrs = json.load(zj)['attributes']        

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

        image_array = self.store[str(resolution)]

        if stack and channel:
            s_idx = self._label_to_index('stack', stack)
            c_idx = self._label_to_index('channel', channel)
            return image_array[s_idx:s_idx+1,c_idx:c_idx+1,:,:,:]
        elif stack:
            s_idx = self._label_to_index('stack', stack)
            return image_array[s_idx:s_idx+1,:,:,:,:]
        elif channel:
            c_idx = self._label_to_index('channel', channel)
            return image_array[:,c_idx:c_idx+1,:,:,:]

        return image_array
    
    def _label_to_index(self, filter:str, label:str):
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

    def save(self, resolution:str, stack:str=None, channel:str=None):
        pass
