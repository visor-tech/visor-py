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

# Open array of resolution 0
# arr is a zarr.Array with 5-dimensions: vs,ch,z,y,x
# see more about array format at https://visor-tech.github.io/visor-data-schema
arr = v_img_r.open(resolution=0)

# Convert to numpy.ndarray
# below code loads the entire zarr.Array into memory as a numpy.ndarray
# Note:
#   zarr.Array supports lazy loading and is recommended for large arrays.
np_arr = arr[:]

# Or slice zarr.Array like numpy
# sub_arr is a numpy.ndarray with 5-dimensions: vs=1,ch=1,z,y,x
# Note:
#   The zarrs.py module requires dimensions to be preserved.
#   This operation also loads array into memory.
sub_np_arr = arr[:1,:1,:,:,:]

# To get index by visor_stack(vs) or channel(ch) labels
s1_idx = v_img_r.label_to_index('stack', 'stack_1') # 0
s1_arr = arr[s1_idx:s1_idx+1,:,:,:,:]

c488_idx = v_img_r.label_to_index('channel', '488') # 1
c488_arr = arr[:,c488_idx:c488_idx+1,:,:,:]

# Convert to dask.array.Array
# below code converts a zarr.Array to a dask.array.Array
# but will not load data into memory just yet
# Note:
#   We like Dask but do not recommend it for disk writing with Zarr v3 (sharded Zarr) for now, because it relies on zarr.py, which has 10× slower disk writing performance compared to zarrs.py.
import dask.array as da
da_arr = da.from_array(arr, chunks=arr.chunks)

# ------

# Create Image
new_path = 'path/to/VISOR002.vsr'
## Construct writable Image, fail if exists
## v_img_w is an instance of visor.Image
v_img_w = visor.Image(
    new_path,
    type='raw',
    name='slice_1_10x',
    mode='w-',
)

## Metadata
##   follow https://visor-tech.github.io/visor-data-schema/
attr = {...}    # .vsr/visor_{type}_images/{name}.zarr/zarr.json['attributes']

new_arr_shape      = (2,2,4,4,4)
new_arr_shard_size = (1,1,4,4,4)
new_arr_chunk_size = (1,1,2,2,2)
dtype='uint16'

## Create new zarr array on disk
zarray1 = v_img_w.create(
    resolution='1',
    dtype=dtype,
    attr=attr,
    shape=new_arr_shape,
    shard_size=new_arr_shard_size,
    chunk_size=new_arr_chunk_size,
    compressors=BloscCodec(cname="zstd", clevel=5),
)

## Generate a random array
import numpy as np
new_arr = np.random.randint(0, 255, size=new_arr_shape, dtype=dtype)

## Save numpy.ndarray to zarr array on disk
zarray1[...] = new_arr

# Save to an existing zarr array on disk
zarray0 = v_img_w.open(resolution='0')
zarray0[:1,:,:,:,:] = new_arr[:1,:,:,:,:]

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
