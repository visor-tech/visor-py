from pathlib import Path
import zarr
import zarrs
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

def load_image(path:str|Path, type:str, name:str, resolution:str,
               stack=None, channel=None):
    """
    Load Image, as a zarr array

    Parameters:
        path:       path to the .vsr file
        type:       image type, see visor.info()['image_types']
        name:       name of slice, see visor.list_image()
        resolution: resolution level, see visor.list_image()
        stack:      visor stack label
        channel:    channel wavelength

    Returns:
        zarr.Array
    """
    image_path = path/f'visor_{type}_images'/f'{name}.zarr'
    zarr_store = zarr.open(image_path, mode='r')
    zarray = zarr_store[resolution]

    return zarray
