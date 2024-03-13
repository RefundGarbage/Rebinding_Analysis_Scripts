import numpy as np
import pandas as pd
from skimage import io as imgio
import os
from natsort import natsorted
import time
import tifffile
import tomllib

def main():

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config_path = os.path.join(__location__, 'script-config.toml')
    with open(config_path, 'rb') as config_file:
        configs = tomllib.load(config_file)

    csv_path = configs['path']['csv_path']
    mask_path = configs['path']['mask_path']


    outpath = csv_path + "\\_ColBD_LIFE_FInalN2"

    enable_fixed_particle = False
    use_gap_fixes = True
    particle_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\particles_result'

    max_frame = 50000

    colors = {
        'Cell_Background': [0, 0, 0],
        'Cell_Border': [25,25,25],
        'Bound_Center': [171, 122, 235],
        'Bound_Outer': [171, 122, 235],
        'Diffuse_Center': [250, 121, 116],
        'Diffuse_Outer': [250, 121, 116],
        'Constricted_Center': [76, 110, 168],
        'Constricted_Outer': [76, 110, 168],
        'Fixed-Particle': [133, 212, 154],
        'Gap': [102, 255, 102]
    }

    masks = natsorted(get_file_names_with_ext(mask_path, 'png'))
    outlines = natsorted(get_file_names_with_ext(mask_path, 'txt'))
    tracks = pd.read_csv((outpath + "\\_ColBD_LIFE_bound_decisions.csv") if not use_gap_fixes else
                         (outpath + "\\_ColBD_LIFE_gaps-and-fixes_decisions.csv"))
    tracks = tracks.loc[:, ~tracks.columns.str.contains('^Unnamed')]

    if enable_fixed_particle:
        particles_files = natsorted(get_file_names_with_ext(particle_path, 'csv'))

    headers = tracks[['Video #', 'Cell', 'Track']].to_numpy()
    tracks = slice_tracks(tracks, headers)
    tracks_by_video = [[] for i in range(len(masks))]
    for track in tracks:
        tracks_by_video[track.iloc[0]['Video #'] - 1].append(track)

    for i in range(len(masks)):
        print('(Video ' + str(i+1) +') -> Mask: ' + masks[i])
        mask = np.swapaxes(imgio.imread(masks[i]), 0, 1)
        if enable_fixed_particle:
            particles = pd.read_csv(particles_files[i]).to_numpy()
        outline = []
        with open(outlines[i], 'r') as file:
            for line in file:
                outline.append(np.array(line.split(',')).astype(int))
        video = inintialize_video(mask, outline, max_frame, colors['Cell_Background'], colors['Cell_Border'], enable_fixed_particle)
        if enable_fixed_particle:
            video = parse_fixed_spots(video, particles, max_frame, i+1, colors['Fixed-Particle'])
        video = parse_tracks(video, tracks_by_video[i], 'Bound', colors)

        video = np.swapaxes(video, 1, 2).astype('uint8')
        save_path = outpath + '\\' + str(i+1) + 'b' + ".tif"
        print('\t-> Saved to:', save_path)
        tifffile.imwrite(save_path,
                         video, imagej=True, photometric='rgb', metadata={'axes': 'TYXS', 'mode': 'composite'})
    return

def parse_tracks(video, tracks_all, key, colors):
    for tracks in tracks_all:
        record = (int(tracks.iloc[0]['x']), int(tracks.iloc[0]['y']))
        for iter in range(len(tracks.index)):
            spot = tracks.iloc[iter]
            mark = spot[key]
            frame, x, y = int(spot['Frame']), int(np.round(spot['x'])), int(np.round(spot['y']))
            gap = x == -1 or y == -1
            if(gap):
                x, y = record
            else:
                record = (int(np.round(spot['x'])), int(np.round(spot['y'])))
            if(frame >= video.shape[0]): continue # not really necessary
            outer = spot_index(x, y)
            video[frame][x][y] = (
                colors['Gap'].copy()) if gap else (
                colors['Diffuse_Center'].copy()) if mark == 0 else (
                colors['Constricted_Center'].copy()) if mark == 1 else (
                colors['Bound_Center'].copy())
            for x1, y1 in outer:
                try:
                    video[frame][x1][y1] = (
                        colors['Gap'].copy()) if gap else (
                        colors['Diffuse_Outer'].copy()) if mark == 0 else (
                        colors['Constricted_Outer'].copy()) if mark == 1 else (
                        colors['Bound_Outer'].copy())
                except IndexError:
                    continue
    return video

def parse_fixed_spots(video, particles, max_frame, vid_index, color_particle):
    for iter in range(len(particles)):
        particle = particles[iter]
        nx, ny, xx, xy = particle[7:11].astype(int).tolist()
        for i in np.arange(nx, xx + 1, 1):
            for j in np.arange(ny, xy + 1, 1):
                video[i][j] = color_particle.copy()
    video = np.repeat(video[np.newaxis, :, :, :], max_frame, axis=0)
    return video

# Video is going to be a width by height by length array with rgb color
def inintialize_video(mask, outline, max_frame, cell_color, cell_border_color, use_fixed):
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
    if not use_fixed: video = np.repeat(video[np.newaxis, :, :, :], max_frame, axis=0)
    return video

def slice_tracks(tracks, headers):
    indices = []
    save = np.array([-1, -1, -1])
    for i in range(headers.shape[0]):
        if not np.all(headers[i] == save):
            save = headers[i].copy()
            indices.append(i)
    indices.append(headers.shape[0])

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