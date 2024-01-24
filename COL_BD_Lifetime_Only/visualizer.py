import numpy as np
import pandas as pd
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
    csv_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\Tracking'  # csv from trackmate
    mask_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\_seg'  # *.png

    max_frame = 2000

    colors = {
        'Cell_Background': [211, 211, 211],
        'Cell_Border': [168, 250, 35],
        'Bound_Center': [121, 29, 242],
        'Bound_Outer': [171, 122, 235],
        'Diffuse_Center': [250, 57, 50],
        'Diffuse_Outer': [250, 121, 116],
        'Constricted_Center': [2, 60, 161],
        'Constricted_Outer': [76, 110, 168]
    }

    masks = natsorted(get_file_names_with_ext(mask_path, 'png'))
    outlines = natsorted(get_file_names_with_ext(mask_path, 'txt'))
    outpath = csv_path + "\\_ColBD_LIFE"
    tracks = pd.read_csv(outpath + "\\_ColBD_LIFE_bound_decisions.csv")
    tracks = tracks.loc[:, ~tracks.columns.str.contains('^Unnamed')]

    headers = tracks[['Video #', 'Cell', 'Track']].to_numpy()
    tracks = slice_tracks(tracks, headers)
    tracks_by_video = [[] for i in range(len(masks))]
    for track in tracks:
        tracks_by_video[track.iloc[0]['Video #'] - 1].append(track)

    for i in range(len(masks)):
        print('(Video ' + str(i+1) +') -> Mask: ' + masks[i])
        mask = np.swapaxes(imgio.imread(masks[i]), 0, 1)
        outline = []
        with open(outlines[i], 'r') as file:
            for line in file:
                outline.append(np.array(line.split(',')).astype(int))
        video = inintialize_video(mask, outline, max_frame, colors['Cell_Background'], colors['Cell_Border'])
        video = parse_tracks(video, tracks_by_video[i], 'Bound', colors)

        video = np.swapaxes(video, 1, 2).astype('uint8')
        save_path = outpath + '\\' + str(i+1) + 'b' + ".tif"
        print('\t-> Saved to:', save_path)
        tifffile.imwrite(save_path,
                         video, imagej=True, photometric='rgb', metadata={'axes': 'TYXS', 'mode': 'composite'})
    return

def parse_tracks(video, tracks_all, key, colors):
    for tracks in tracks_all:
        for iter in range(len(tracks.index)):
            spot = tracks.iloc[iter]
            mark = spot[key]
            frame, x, y = int(spot['Frame']), int(np.round(spot['x'])), int(np.round(spot['y']))
            if(frame >= video.shape[0]): continue # not really necessary
            outer = spot_index(x, y)
            video[frame][x][y] = colors['Diffuse_Center'].copy() if mark == 0 else colors['Constricted_Center'].copy() if mark == 1 else colors['Bound_Center'].copy()
            for x1, y1 in outer:
                try:
                    video[frame][x1][y1] = colors['Diffuse_Outer'].copy() if mark == 0 else colors['Constricted_Outer'].copy() if mark == 1 else colors['Bound_Outer'].copy()

                except IndexError:
                    continue
    return video

# Video is going to be a width by height by length array with rgb color
def inintialize_video(mask, outline, max_frame, cell_color, cell_border_color):
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
    video = np.repeat(video[np.newaxis, :, :, :], max_frame, axis=0)
    return video

def slice_tracks(tracks, headers):
    indices = []
    save = np.array([-1, -1, -1])
    for i in range(headers.shape[0]):
        if not np.all(headers[i] == save):
            save = headers[i].copy()
            indices.append(i)

    tracks_sliced = []
    for i in range(len(indices) - 1):
        tracks_sliced.append(tracks.iloc[indices[i] : indices[i+1], :])
    return tracks_sliced

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