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
import visor.image as vimg
```

#### Examples
- Read .vsr
```py
import visor.image as vimg

# Open .vsr file with read-only ('r') mode
# img is a visor.Image object 
img = vimg.open_vsr('path/to/VISOR001.vsr', 'r')
print(img.info)
```

- Read raw image
```py
import visor.image as vimg
img = vimg.open_vsr('path/to/VISOR001.vsr', 'r')

# Read raw image of slice_1_10x from zarr file
# arr is a visor.Array object
arr = img.read(img_type='raw',
               zarr_file='slice_1_10x.zarr',
               resolution=0)
print(arr.info)
```

- Read arrays by named visor_stacks or channels
```py
import visor.image as vimg
img = vimg.open_vsr('path/to/VISOR001.vsr', 'r')
arr = img.read(img_type='raw',
               zarr_file='slice_1_10x.zarr',
               resolution=0)

# Read visor_stacks by label
# s1_array is a numpy.ndarray, dimensions: vs=1,ch,z,y,x
s1_array = arr.read(stack='stack_1')
print(f's1_array shape: {s1_array.shape}, dimensions: vs=1,ch,z,y,x')

# Read channel by wavelength
# c488_array is a numpy.ndarray, dimensions: vs,ch=1,z,y,x
c488_array = arr.read(channel='488')
print(f'c488_array shape: {c488_array.shape}, dimensions: vs,ch=1,z,y,x')

# Read visor_stack and channel
# s1c488_array is a numpy.ndarray, dimensions: vs=1,ch=1,z,y,x
s1c488_array = arr.read(stack='stack_1', channel='488')
print(f's1c488_array shape: {s1c488_array.shape}, dimensions: vs=1,ch=1,z,y,x')
```

- Read array by index
```py
import visor.image as vimg
img = vimg.open_vsr('path/to/VISOR001.vsr', 'r')
arr = img.read(img_type='raw',
               zarr_file='slice_1_10x.zarr',
               resolution=0)

# Read by index
# the_array is a numpy.ndarray
the_array = arr.array
print(f'the_array shape: {the_array.shape}, dimensions: vs,ch,z,y,x')
# subarr is a numpy.ndarray
subarr = the_array[0,0,:,:,:]
print(f'subarr shape: {subarr.shape}, dimensions: z,y,x')
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
