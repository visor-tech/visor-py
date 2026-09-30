from pathlib import Path
from .image import Image

class ROI:

    def __init__(self, image_path:str|Path=None,
        resolution:str|int=None,
        ranges:tuple[slice|int, ...]=None,
        image:Image=None):
        """
        Constructor of ROI

        Parameters:
            image_path: path of image (alternative to image)
            resolution: resolution level
            ranges:     tuple of roi ranges (int or slice) in each dimension
            image:      a visor.Image instance (alternative to image_path)
        """
        if image is not None:
            if resolution is None:
                raise ValueError('ROI requires a resolution level.')
            self.img = image
        else:
            if image_path is None:
                raise ValueError('ROI requires image_path or image.')
            img_path = Path(image_path)
            self.img = Image(
                img_path.parent.parent,
                image_type=img_path.parent.name.split('_')[1],
                image_name=img_path.name.replace('.zarr',''),
            )
        self.resolution = resolution
        self.ranges = ranges


    def load(self):
        """
        Load ROI array

        Returns:
            numpy.ndarray
        """
        return self.img.load(self.resolution)[self.ranges]