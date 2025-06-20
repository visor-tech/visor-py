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
- Check VSR Info / List Content
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
## Filter transform list
##   get version from visor.info
visor.list_transform(path, version='xxx_20250525')
```

- Work with Image
```py
import visor
path = 'path/to/VISOR001.vsr'

# Construct read_only Image from raw image of slice_1_10x from zarr file
# v_img_r is an instance of visor.Image
v_img_r = visor.Image(
    path,
    type='raw',
    name='slice_1_10x',
    mode='r',
)

# Load array of resolution 0
# arr is a zarr.Array with 5-dimensions: vs,ch,z,y,x
# see more about array format at https://visor-tech.github.io/visor-data-schema
# Note: arr can also be loaded by label-filters, see below
arr = v_img_r.load(resolution=0)

# Convert to numpy.ndarray
# below code loads the entire zarr.Array into memory as a numpy.ndarray
# Note: zarr.Array and dask.array.Array are lazy loading, 
#       that are recommended for large arrays
np_arr = arr[:]

# Or slice zarr.Array like numpy
# sub_arr is a numpy.ndarray with 3-dimensions: z,y,x
sub_np_arr = arr[1,1,:,:,:]

# Load array with visor_stack filter by label 'stack_1'
# s1_arr is a numpy.ndarray with 5-dimensions: vs=1,ch,z,y,x
s1_arr = v_img_r.load_stack(
    resolution=0,
    stack='stack_1'
)

# Load array with channel filter by label '488'
# c488_arr is a numpy.ndarray with 5-dimensions: vs,ch=1,z,y,x
c488_arr = v_img_r.load_channel(
    resolution=0,
    channel='488'
)

# Load array of visor_stack and channel filter by label
# s1c488_arr is a numpy.ndarray with 5-dimensions: vs=1,ch=1,z,y,x
s1c488_arr = v_img_r.load_stack_channel(
    resolution=0,
    stack='stack_1',
    channel='488'
)

# Convert to dask.array.Array
# below code converts a zarr.Array to a dask.array.Array
# but will not load data into memory just yet
import dask.array as da
da_arr = da.from_array(arr, chunks=arr.chunks)

# ------

# Create Image
new_path = 'path/to/VISOR002.vsr'

## Metadata
##   follow https://visor-tech.github.io/visor-data-schema/
info = {...}    # .vsr/info.json
attr = {...}    # .vsr/visor_{type}_images/{name}.zarr/zarr.json['attributes']

## Generate a random array
## new_arr is a numpy.ndarray
new_arr_shape      = (2,2,4,4,4)
new_arr_shard_size = (1,1,4,4,4)
new_arr_chunk_size = (1,1,2,2,2)

import numpy as np
new_arr = np.random.randint(0, 255, size=new_arr_shape, dtype='uint16')

## Construct writable Image, fail if exists
## v_img_w is an instance of visor.Image
v_img_w = visor.Image(
    new_path,
    type='raw',
    name='slice_1_10x',
    mode='w-',
    resolution='0',
    info=info,
    attr=attr,
    shape=new_arr_shape,
    shard_size=new_arr_shard_size,
    chunk_size=new_arr_chunk_size,
)
v_img_w.save(new_arr)

# Save a visor_stack
# Note: we preserve the visor_stack(vs) dimension here for consistency
s1_new_arr = new_arr[:1,:,:,:,:]
v_img_w.save(s1_new_arr, stack='stack_1')

# Save a channel, similarly
c488_new_arr = new_arr[:,:1,:,:,:]
v_img_w.save(c488_new_arr, channel='488')
```

- Work with Transform
```py
import visor
path = 'path/to/VISOR001.vsr'

# Construct Transform with version and name
# v_xfm is an instance of visor.Transform
v_xfm = visor.Transform(
    path,
    version='xxx_20250525',
    name='slice_1_10x',
)

# Save slice_1_10x's raw_to_ortho affine transform as zarr
# Note: transform type and format could be various
# see more about transform type / format at https://visor-tech.github.io/visor-data-schema
raw_to_ortho_mat = [0, np.sin(45) * 1.03, 0,
                    0, 0, 1.03,
                    3.5, np.cos(45) * 1.03, 0]
v_xfm.save(
    raw_to_ortho_mat,
    name='raw_to_ortho',
    type='affine',
    format='zarr',
)

# Resample
# roi is a tuple of slices
#   - region in source space ('raw')
# rs_arr is a numpy.ndarray
#   - resampled array in target space ('ortho')
roi = (slice(1),slice(1),slice(16,32),slice(16,32),slice(16,32))
rs_arr = v_xfm.resample(
    roi,
    from_space='raw',
    to_space='ortho',
)
```

## References
[VISoR Image Schema](https://visor-tech.github.io/visor-data-schema)
