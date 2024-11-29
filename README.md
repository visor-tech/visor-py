# visor-py
A Python Library for VISoR Image.

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

- Read arrays by named stacks or channels
```py
import visor.image as vimg
img = vimg.open_vsr('path/to/VISOR001.vsr', 'r')
arr = img.read(img_type='raw',
               zarr_file='slice_1_10x.zarr',
               resolution=0)

# Read stack by label
# s1_array is a dask.array, dimensions: c,z,y,x
s1_array = arr.read(stack='stack_1')
print(f's1_array shape: {s1_array.shape}, dimensions: c,z,y,x')

# Read channel by wavelength
# c488_array is a dask.array, dimensions: s,z,y,x
c488_array = arr.read(channel='488')
print(f'c488_array shape: {c488_array.shape}, dimensions: s,z,y,x')

# Read stack and channel
# s1c488_array is a dask.array, dimensions: z,y,x
s1c488_array = arr.read(stack='stack_1', channel='488')
print(f's1c488_array shape: {s1c488_array.shape}, dimensions: z,y,x')
```

- Read array by index
```py
import visor.image as vimg
img = vimg.open_vsr('path/to/VISOR001.vsr', 'r')
arr = img.read(img_type='raw',
               zarr_file='slice_1_10x.zarr',
               resolution=0)

# Read by index
# the_array is a dask.array
the_array = arr.array
print(f'the_array shape: {the_array.shape}, dimensions: s,c,z,y,x')
# subarr is a dask.array
subarr = the_array[0,0,:,:,:]
print(f'subarr shape: {subarr.shape}, dimensions: z,y,x')
```

- Write .vsr
```py
import visor.image as vimg
import numpy as np
import dask.array as da

# Open .vsr file with write ('w') mode
# img is a visor.Image object 
img = vimg.open_vsr('path/to/VISOR001.vsr', 'w')

# Generate a random array
# new_arr is a dask array
new_arr = da.random.random(size=(2, 2, 4, 4, 4), chunks=(1, 1, 2, 2, 2))

# Metadata
# follow https://visor-tech.github.io/visor-data-schema/
img_info = {}    # info.json
arr_info = {}    # .zattrs
selected = {}    # selected.json

# Write array to .vsr
img.write(arr=new_arr,
          img_type='raw',
          file='slice_1_10x',
          resolution=0,
          img_info=img_info,
          arr_info=arr_info,
          selected=selected)
```

## References
[VISoR Image Schema](https://visor-tech.github.io/visor-data-schema)
