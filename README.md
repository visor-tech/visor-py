# visor-py
A Python Library for VISoR Image.

> [!NOTE]
> Since [v2025.5.1](https://github.com/visor-tech/visor-py/releases/tag/v2025.5.1), we've switched to [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html), are using [zarrs](https://github.com/ilan-gold/zarrs-python) to speed up I/O, and have replaced [dask](https://github.com/dask/dask) since it has not yet optimized its I/O for sharded Zarr.

## Usage
#### Install Module
```sh
pip install visor-py
```

#### Import Module
```py
import visor
```

#### Examples
- Check VSR Info / Content
```py
import visor
path = 'path/to/VISOR001.vsr'

# Check vsr info
visor.info(path)

# List image
visor.list_image(path)
## Filter image list
##   get type/channel from visor.info
visor.list_image(path, type='raw')
visor.list_image(path, channel='405')
visor.list_image(path, type='raw', channel='405')

# List transform
visor.list_transform(path)
```

- Read Image
```py
import visor
path = 'path/to/VISOR001.vsr'

# Read raw image of slice_1_10x from zarr file
# arr is a zarr.Array with 5-dimensions: vs,ch,z,y,x
# see more on array format at https://visor-tech.github.io/visor-data-schema
arr = visor.read(path
  type='raw',
  name='slice_1_10x',
  resolution=0
)

# Read visor_stack image by label
# s1_arr is a zarr.Array with 5-dimensions: vs=1,ch,z,y,x
s1_arr = visor.read(path
  type='raw',
  name='slice_1_10x',
  resolution=0,
  stack='stack_1'
)

# Read channel by wavelength by label
# c488_arr is a zarr.Array with 5-dimensions: vs,ch=1,z,y,x
c488_arr = visor.read(path
  type='raw',
  name='slice_1_10x',
  resolution=0,
  channel='488'
)

# Read visor_stack and channel by label
# s1c488_arr is a zarr.Array with 5-dimensions: vs=1,ch=1,z,y,x
s1c488_arr = visor.read(path
  type='raw',
  name='slice_1_10x',
  resolution=0,
  stack='stack_1',
  channel='488'
)
```

- Use numpy or dask
```py
import visor
import numpy as np
import dask.array as da

# First read out zarr.Array
arr = visor.read('path/to/VISOR001.vsr',
  type='raw',
  name='slice_1_10x',
  resolution=0
)

# Convert to numpy.ndarray
# below code reads the entire zarr.Array into memory as a numpy.ndarray
# Note: zarr.Array and dask.array are lazy loading, 
#       that are recommended for large arrays
np_arr = arr[:]

# Convert to dask.array
# below code convert a zarr.Array to a dask.array
da_arr = da.from_array(arr, chunks=arr.chunks)
```

- Write .vsr
```py
import visor.image as vimg
import numpy as np

# Open .vsr file with write ('w') mode
# img is a visor.Image object 
img = vimg.open_vsr('path/to/VISOR001.vsr', 'w')

# Generate a random array
# new_arr is a numpy.ndarray
new_arr_shape      = (2,2,4,4,4)
new_arr_shard_size = (1,1,4,4,4)
new_arr_chunk_size = (1,1,2,2,2)

new_arr = np.random.randint(0, 255, size=new_arr_shape, dtype='uint16')

# Metadata
# follow https://visor-tech.github.io/visor-data-schema/
img_info = {}    # info.json
arr_info = {}    # zarr.json['attributes']
selected = {}    # selected.json

# Write array to .vsr
img.write(arr=new_arr,
          img_type='raw',
          file='slice_1_10x',
          resolution=0,
          img_info=img_info,
          arr_info=arr_info,
          chunk_size=new_arr_chunk_size,
          shard_size=new_arr_shard_size,
          selected=selected)
```

## References
[VISoR Image Schema](https://visor-tech.github.io/visor-data-schema)
