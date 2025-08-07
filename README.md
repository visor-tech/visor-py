# visor-py
A Python Library for VISoR Image.

> [!NOTE]
> Since [v2025.5.1](https://github.com/visor-tech/visor-py/releases/tag/v2025.5.1), we've switched to [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html), are using [zarrs](https://github.com/ilan-gold/zarrs-python) to speed up I/O, and have replaced [dask](https://github.com/dask/dask) since it has not yet optimized its I/O for sharded Zarr.

# Usage
### Install Module
```sh
pip install visor-py
```

### Import Module
```py
import visor
```

### Examples
#### VSR
- Construct VSR
```py
vsr_path = 'path/to/VISOR001.vsr'
vsr:visor.VSR = visor.VSR(vsr_path)
```

- Check vsr info
```py
vsr.info() # return JSON
```

- List images
```py
vsr.images() # return JSON
# Filter images by image_type
#   get image_types from vsr.info()
vsr.images(image_type='raw')
```

- List transforms
```py
vsr.transforms() # return JSON
# Filter transforms by recon_version
#   get recon_versions from vsr.info()
vsr.transforms(recon_version='xxx_20250525')
```

- Create vsr if not exist
```py
new_vsr_path = 'path/to/VISOR002.vsr'
vsr = visor.VSR(new_vsr_path, create=True)
```

#### Image
- Construct Image
```py
vsr_path = 'path/to/VISOR001.vsr'
# v_img is an instance of visor.Image
v_img:visor.Image = visor.Image(
    vsr_path,
    image_type='raw',
    image_name='slice_1_10x',
)
```

- Load array of resolution 0
```py
# arr is a zarr.Array with 5-dimensions: vs,ch,z,y,x
# see more about array format at https://visor-tech.github.io/visor-data-schema
arr:zarr.Array = v_img.load(resolution='0')
```

- Convert to numpy.ndarray
```py
# below code loads the entire zarr.Array into memory as a numpy.ndarray
# Note:
#   zarr.Array supports lazy loading and is recommended for large arrays.
np_arr:numpy.ndarray = arr[:]
```

- Or slice zarr.Array like numpy
```py
# sub_arr is a numpy.ndarray with 5-dimensions: vs=1,ch=1,z,y,x
# Note:
#   The zarrs.py module requires dimensions to be preserved.
#   This operation also loads array into memory.
sub_np_arr:numpy.ndarray = arr[:1,:1,:,:,:]

# To get index by visor_stack(vs) or channel(ch) labels
s1_idx = v_img.label_to_index('stack', 'stack_1') # 0
s1_arr = arr[s1_idx:s1_idx+1,:,:,:,:]

c488_idx = v_img.label_to_index('channel', '488') # 1
c488_arr = arr[:,c488_idx:c488_idx+1,:,:,:]
```

- Convert to dask.array.Array
```py
# below code converts a zarr.Array to a dask.array.Array
# but will not load data into memory just yet
# Note:
#   We like Dask but do not recommend it for disk writing with Zarr v3 (sharded Zarr) for now, because it relies on zarr.py, which has 10× slower disk writing performance compared to zarrs.py.
import dask.array as da
da_arr = da.from_array(arr, chunks=arr.chunks)
```

- Create Image
```py
v_img = visor.Image(
    vsr_path,
    image_type='raw',
    image_name='slice_2_10x',
    create=True,
)
# Metadata
#   follow https://visor-tech.github.io/visor-data-schema/
#   .vsr/visor_{image_type}_images/{image_name}.zarr/zarr.json['attributes']
attrs = {...}
v_img.update_attrs(attrs)
# Generate a random array
new_arr_shape      = (2,2,4,4,4)
new_arr_shard_size = (1,1,4,4,4)
new_arr_chunk_size = (1,1,2,2,2)
dtype='uint16'
new_arr = numpy.random.randint(0, 255, size=new_arr_shape, dtype=dtype)
# Save array to disk
v_img.save(
    new_arr,
    resolution='0',
    dtype=dtype,
    shape=new_arr_shape,
    shard_size=new_arr_shard_size,
    chunk_size=new_arr_chunk_size,
    compressors=BloscCodec(cname="zstd", clevel=5),
)
```

- Modify Image
```py
# Update partial data to an existing zarr array on disk
# where 
#   arr is a zarr.Array
#   new_arr is a numpy.ndarray
v_img = visor.Image(
    vsr_path,
    image_type='raw',
    image_name='slice_2_10x',
    create=True,
)
arr = v_img.load(resolution='0')
arr[:1,:,:,:,:] = new_arr[:1,:,:,:,:]

# Update metadata
attrs = v_img.attrs
attrs['visor']['visor_stacks'].append(
    {
        "index": 1,
        "label": "stack_2",
        "position": [20.2647, 63.2581]
    }
)
v_img.update_attrs(attrs)
```

#### ROI
- Construct and Load ROI
```py
# v_roi is an instance of visor.ROI
v_roi:visor.ROI = visor.ROI(
    image_path=v_img.path, # :str|Path
    resolution='0',        # :str|int
    ranges=(1,1,slice(2,3),slice(None),slice(None)),
    # :tuple[slice|int, ...], ch,st,z,y,x
)
# np_arr is a numpy.ndarray
np_arr:numpy.ndarray = v_roi.load()
```

#### Transform
- Construct Transform
```py
# v_xfm is an instance of visor.Transform
v_xfm = visor.Transform(
    vsr_path,
    recon_version='xxx_20250525',
    slice_name='slice_1_10x',
)
```

- Load Transform
```py
# Function load() could be polymorphic based on transform type (e.g. affine) and file format (e.g. tfm), for example
# raw_to_ortho is an affine transform stored as SimpleITK tfm format
#   below load transform for stack index 0 and channel index 0
#   where indices cooresponding to raw image
# t_raw_to_ortho is an instance of SimpleITK.Transform
t_raw_to_ortho = v_xfm.load(
    from_space='raw',
    to_space='ortho',
    params=[0,0],
)
```

- Create Transform
```py
v_xfm = visor.Transform(
    vsr_path,
    recon_version='xxx_20250525',
    slice_name='slice_1_10x',
    create=True,
)

raw_to_ortho_mat = [0, np.sin(45) * 1.03, 0,
                    0, 0, 1.03,
                    3.5, np.cos(45) * 1.03, 0]

offset_vec = [0,0,0]
stack_idx = 0
channel_idx = 0
params = [stack_idx] + [channel_idx] + raw_to_ortho_mat + offset_vec,
# another example could be
# params = [stack_idx] + [channel_idx] + model_params,

# Save transform to disk
# PATH: version/slice_/space_to_space/stack/channel/type.format
# see more about transform type / format at https://visor-tech.github.io/visor-data-schema
t_type = 'affine'
t_format = 'tfm'
v_xfm.save(
    from_space='raw',
    to_space='ortho',
    t_type=t_type,
    t_format=t_format,
    params=params,
)
v_xfm.update_meta(
    trans = {
        'name'  : f'{from_space}_to_{to_space}',
        'type'  : t_type,
        'format': t_format,
    }
)
```

# References
[VISoR Image Schema](https://visor-tech.github.io/visor-data-schema)
