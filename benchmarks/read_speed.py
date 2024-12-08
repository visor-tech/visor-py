# Benchmark the performance of the visor reader

# Usage:
# python read_speed.py <bench_mode> <image_path>

# Example:
# python read_speed.py full-sum /share/data/VISoR_Data/zarr/N1779.vsr

import os
import sys
import time
import numpy as np
import dask

import visor.image as vimg

def get_total_size(directory):
    total_size = 0
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
            n_files += 1
    return (total_size, n_files)

def loop_all_stacks(visor_img, res, func):
    li = visor_img.list('raw')
    n_ch_set = [len(it['channels']) for it in li]
    if np.max(n_ch_set) == np.min(n_ch_set):
        n_channels = n_ch_set[0]
    else:
        n_channels = (np.min(n_ch_set), np.max(n_ch_set))
    stats = {
        "n_slices": len(li),
        "n_channels": n_channels,
        "n_stacks": 0,
        "n_frames": 0,
        "n_pixels": 0,
        "n_bytes": 0,
    }
    s = []
    for slice_info in visor_img.list('raw'):
        slice_path = slice_info['path']
        slice_ch = slice_info['channels']
        print(f'-- Slice "{slice_path}".')
        img = visor_img.read('raw', slice_path, res)
        for stack_name in img.stack_map:
            print(f'  -- Stack "{stack_name}".')
            for ch in img.channel_map:
                print('    -- ch ', ch)
                da = img.read(stack=stack_name, channel=ch)
                # print(da.shape, da.dtype, da.size, da.chunks)
                stats['n_pixels'] += da.size
                stats['n_frames'] += da.shape[0]
                stats['n_stacks'] += 1
                stats['n_bytes'] += da.nbytes
                if func is not None:
                    s.append(func(da))
                else:
                    s.append(None)
    if func is not None:
        s = dask.compute(s)[0]
        #print(type(s[0]))
    return s, stats

def benchmark_reader(visor_img_path, res, bench_mode):
    print(f'Dataset "{visor_img_path}"')
    t0 = time.time()
    visor_img = vimg.open_vsr(visor_img_path, 'r')
    t1 = time.time()
    print(f'Time Init: {t1-t0:.3f}s')
    if bench_mode == 'full-none':
        ret, stats = loop_all_stacks(visor_img, res, None)
        ans_st = 'None'
    elif bench_mode == 'full-sum':
        func = lambda da: da.sum()
        ret, stats = loop_all_stacks(visor_img, res, func)
        ans_st = f'sum {np.array(ret).sum()}'
    t2 = time.time()
    img_disk_size, n_files = get_total_size(visor_img_path)
    print('')
    print('Summary')
    print('=======')
    print(f'Image file: "{visor_img_path}"')
    print('Basic stat:')
    print(f'  n_slices   = {stats["n_slices"]}')
    print(f'  n_stacks   = {stats["n_stacks"]}')
    print(f'  n_channels = {stats["n_channels"]}')
    print(f'  n_frames   = {stats["n_frames"]}')
    print(f'  n_pixels   = {stats["n_pixels"]}')
    print('File stat:')
    print(f'  img_disk_size  = {img_disk_size/2**30:.3g} GBytes')
    print(f'  n_files        = {n_files}')
    print(f'  MByte per file = {img_disk_size/n_files/2**20:.3g} MBytes')
    print(f'Result of "{bench_mode}" = {ans_st}')
    wt = t2-t1
    print(f'Time costs:')
    print(f'  {wt:.1f} s for whole sample')
    print(f'  {wt/stats["n_stacks"]:.3g} s per stack')
    print(f'Speed:')
    print(f'  {stats["n_frames"]/wt:.3g} frames per second')
    print(f'  {stats["n_bytes"]/2**20/wt:.3g} MBytes(uncompressed) per second')
    print(f'  {img_disk_size/2**20/wt:.3g} MBytes(on disk) per second')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Test reading speed of visor image")
        print("Usage: python read_speed.py <bench_mode> <visor_image_path>")
        print("  bench_mode: full-none or full-sum")
        sys.exit(1)

    bench_mode = sys.argv[1]
    visor_img_path = sys.argv[2]

    res = 0     # use highest resolution
    benchmark_reader(visor_img_path, res, bench_mode)