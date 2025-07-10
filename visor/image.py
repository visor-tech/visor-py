from pathlib import Path
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
from zarr.codecs import BytesCodec
import numpy

class Image:

    def __init__(self, vsr_path:str|Path,
                 image_type:str, image_name:str, create=False):
        """
        Constructor of Image

        Parameters:
            vsr_path:   path to the .vsr file
            image_type: image type, see vsr.info()['image_types']
            image_name: image name, see vsr.images()
            create:     boolean
        """
        vsr_path = Path(vsr_path)
        # Validate vsr path
        if vsr_path.suffix != '.vsr':
            raise ValueError(f'The path {vsr_path} is not valid, must contain .vsr extension.')
        if not vsr_path.exists() or not vsr_path.is_dir():
            raise NotADirectoryError(f'The path {vsr_path} is not a directory.')
        
        image_path = vsr_path/f'visor_{image_type}_images'/f'{image_name}.zarr'
        if create:
            image_path.mkdir(parents=True, exist_ok=True)
        if not image_path.exists() or not image_path.is_dir():
            raise NotADirectoryError(f'The path {image_path} is not a directory.')

        self.path   = image_path
        self.zgroup = zarr.open_group(image_path)
        self.attrs  = self.zgroup.attrs.asdict()


    def label_to_index(self, filter_type:str, filter_label:str):
        """
        Get index from label of filter stack/channel

        Parameters:
            filter_type:  stack or channel
            filter_label: label of named stack or channel

        Returns:
            int
        """
        v_meta = self.attrs.get('visor')
        if not v_meta:
            raise KeyError("Missing 'visor' metadata in zarr attributes.")

        if 'stack' == filter_type:
            for s in v_meta['visor_stacks']:
                if s['label'] == filter_label:
                    return s['index']
            raise ValueError(f'The visor_stack {filter_label} does not exist.')
        elif 'channel' == filter_type:
            for s in v_meta['channels']:
                if s['wavelength'] == filter_label:
                    return s['index']
            raise ValueError(f'The channel {filter_label} does not exist.')
        else:
            raise ValueError(f'Invalid filter {filter_type}. Must be stack or channel')


    def load(self, resolution:str):
        """
        Load array by resolution

        Parameters:
            resolution: resolution level, see vsr.images()

        Returns:
            zarr.Array
        """

        return self.zgroup[str(resolution)]


    def save(
            self, arr:numpy.ndarray, resolution:str, dtype:str,
            shape:tuple, shard_size:tuple, chunk_size:tuple,
            compressors:BytesCodec):
        """
        Create a zarr array

        Parameters:
            arr:         the array to save
            resolution:  resolution level, see vsr.images()
            dtype:       zarr array dtype
            shape:       zarr array shape
            shard_size:  zarr array shard_size
            chunk_size:  zarr array chunk_size
            compressors: zarr array compressors

        Returns:
            zarr.Array
        """

        array_path = self.path/str(resolution)

        if array_path.is_dir():
            raise FileExistsError(f'The array {array_path} already exist.')
        zarr.create_array(
            store=self.path,
            name=str(resolution),
            dtype=dtype,
            shape=shape,
            shards=shard_size,
            chunks=chunk_size,
            compressors=compressors,
        )

        return self.zgroup[str(resolution)]

    
    def update_attrs(self, attrs:dict):
        """
        Update zarr.json attributes

        Parameters:
            attrs: new attributes
        """  
        self.zgroup.attrs.update(attrs)
