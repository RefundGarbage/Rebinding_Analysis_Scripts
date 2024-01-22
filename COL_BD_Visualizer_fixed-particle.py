import numpy as np
from scipy import io as scio
from skimage import io as imgio
import skimage
import os
from natsort import natsorted
import time
import tifffile
import csv
import logging
import shutil
import sys

def main():
    col_bd_fixed = 'F:\\_Microscopy\\Rawdates\\202301109_TetR_haloQmutnWT\\Images\\COMB\\AnalysisRebindCBC_5quality\\pr208\\_ColBD_fixed-particles.csv'
    col_bd_spots = 'F:\\_Microscopy\\Rawdates\\202301109_TetR_haloQmutnWT\\Images\\COMB\\AnalysisRebindCBC_5quality\\pr208\\_ColBD_spots.csv'
    mask_path = 'F:\\_Microscopy\\Rawdates\\202301109_TetR_haloQmutnWT\\Images\\seg\\pr208\\100ms'  # *.png

    max_frame = 2000

    colors = {
        'Cell_Background': [211, 211, 211],
        'Cell_Border': [168, 250, 35],
        'Bound_Center': [121, 29, 242],
        'Bound_Outer': [171, 122, 235],
        'Diffuse_Center': [250, 57, 50],
        'Diffuse_Outer': [250, 121, 116],
        'Particle': [6, 186, 177]
    }

    masks = natsorted(get_file_names_with_ext(mask_path, 'png'))
    outlines = natsorted(get_file_names_with_ext(mask_path, 'txt'))
    particles = np.loadtxt(col_bd_fixed, delimiter=',', dtype=float)
    tracks = np.loadtxt(col_bd_spots, delimiter=',', dtype=float)
    outpath = "\\".join(col_bd_fixed.split("\\")[:-1]) + "\\_ColBD_Visualizer\\"
    try:
        shutil.rmtree(outpath)
        os.mkdir(outpath)
    except:
        os.mkdir(outpath)

    for i in range(len(masks)):
        print('(Video ' + str(i+1) +') -> Mask: ' + masks[i])
        mask = np.swapaxes(imgio.imread(masks[i]), 0, 1)
        outline = []
        with open(outlines[i], 'r') as file:
            for line in file:
                outline.append(np.array(line.split(',')).astype(int))
        video = inintialize_video(mask, outline, colors['Cell_Background'], colors['Cell_Border'])
        video = parse_fixed_spots(video, particles, max_frame, i+1, colors['Particle']) # comment when no fixed spt
        video = parse_tracks(video, tracks, i+1,
                             colors['Bound_Center'], colors['Bound_Outer'],
                             colors['Diffuse_Center'], colors['Diffuse_Outer'])

        video = np.swapaxes(video, 1, 2).astype('uint8')
        tifffile.imwrite(outpath + str(i+1) + ".tif", video, imagej=True, photometric='rgb', metadata={'axes': 'TYXS', 'mode': 'composite'})
    return

def parse_tracks(video, tracks_all, vid_index, color_bd, color_bd_outer, color_dif, color_dif_outer):
    tracks = slice_for_index(tracks_all, 0, vid_index)
    for iter in range(len(tracks)):
        spot = tracks[iter]
        is_dif = (spot[7] == -1)
        frame, x, y = int(spot[3]), int(np.round(spot[4])), int(np.round(spot[5]))
        if(frame >= video.shape[0]): continue # not really necessary
        outer = spot_index(x, y)
        video[frame][x][y] = color_dif.copy() if is_dif else color_bd.copy()
        for x1, y1 in outer:
            try:
                video[frame][x1][y1] = color_dif_outer.copy() if is_dif else color_bd_outer.copy()
            except IndexError:
                continue
    return video

def parse_fixed_spots(video, particles_all, max_frame, vid_index, color_particle):
    particles = slice_for_index(particles_all, 0, vid_index)
    for iter in range(len(particles)):
        particle = particles[iter]
        nx, ny, xx, xy = particle[8:12].astype(int).tolist()
        for i in np.arange(nx, xx + 1, 1):
            for j in np.arange(ny, xy + 1, 1):
                video[i][j] = color_particle.copy()
    video = np.repeat(video[np.newaxis, :, :, :], max_frame, axis=0)
    return video

# Video is going to be a width by height by length array with rgb color
def inintialize_video(mask, outline, cell_color, cell_border_color):
    video = np.empty(shape=(mask.shape[0], mask.shape[1], 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i][j] == 0:
                video[i][j] = cell_color.copy()

    for i in range(len(outline)):
        x = outline[i][::2]
        y = outline[i][1::2]
        for k in range(len(x)):
            video[x[k]][y[k]] = cell_border_color
    #video = np.repeat(video[np.newaxis, :, :, :], max_frame, axis=0)
    return video

def slice_for_index(array, key, index):
    slice_lower = 0
    slice_upper = 0
    while (slice_upper < array.shape[0] and array[slice_upper][key] <= index):
        if (array[slice_lower][key] < index):
            slice_lower = slice_upper
        slice_upper += 1
    if(slice_upper == array.shape[0] - 1): slice_upper += 1
    return array[slice_lower:slice_upper].copy()

# spot = loc +- 1
def spot_index(x, y):
    return [
        (x+1, y+1),
        (x+1, y),
        (x+1, y-1),
        (x, y+1),
        (x, y-1),
        (x-1, y+1),
        (x-1, y),
        (x-1, y-1)
    ]

'''
================================================================================================================
I/O
================================================================================================================
'''
def get_file_names_with_ext(path:str, ext:str):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fname = file.split('.')
            if(fname[-1] == ext):
                flist.append(root + '\\' +  file)
    return flist

'''
================================================================================================================
START
================================================================================================================
'''

# Start Script
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))