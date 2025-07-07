```py
# generate transform:
# for slice_ in slices:
#     for stack in stacks:
#         for channel in channels:
#             image1 = visor.load_image(slice_, stack_1, channel)
#             image2 = visor.load_image(slice_, stack_2, channel)
#             transform = recon.compute_transform(mode, image1, image2)
#             visor.save_transform(transform, version, slice_, stack, channel)

# resample
# slice_image = visor.resample(
#     vsr_path,
#     recon_version='xxx_20250525',
#     from_space='raw',
#     to_space='ortho',
#     slice_name='slice_1_10x',
#     stack_name='stack_1',
#     channel_name='488',
# )

# for z,y,x in brain:
#     stack     = get_stack(brain)
#     channel   = get_channel(brain)
#     recon_img = resample(stack, channel, image, transform)
# t_slice_1_raw_to_ortho = v_xfm.load('slice_1_10x')


# Apply transform to roi, get roi from
# roi is a tuple of slices
#   - region in source space ('raw')
# rs_arr is a numpy.ndarray
#   - resampled array in target space ('brain')
# points_brain_space = [
#         [ch1, z1,y1,x1],
#         [ch2, z2,y2,x2]
#         ]
# points_raw_space = T_brain_to_raw.apply(points_brain_space)
# assert points_raw_space == [
#                               [slice_idx1, stack_idx1, ch1, z1, y1, x1],
#                               [slice_idx2, stack_idx2, ch2, z2, y2, x2],
#                             ]

# def resample

#     backward_transform = forward_transform.inverse()
#     roi_source = backward_transform.apply(roi_target)
#     target_arr = interpolation(raw_arr, roi_source)
#     return target_arr
```
