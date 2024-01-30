import numpy as np
import pandas as pd
from skimage import io as imgio
import time
import os
from natsort import natsorted
import logging
import shutil


def main():
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA7\\timelapse\\AnalysisRebindCBC_start0_Quality5'  # csv from trackmate
    #mask_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA7\\seg'  # *.png
    csv_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\Tracking'  # csv from trackmate
    mask_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\_seg'  # *.png
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\AnalysisRebindCBC_start0_Quality5' 
    #mask_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\seg'  # *.png


    # Some parameters
    allowed_gap_max = 20
    allowed_track_length_min = 5
    dist_range = 5
    dist_none = float('inf')

    # Output Format
    final_list_track_spots = []
    final_list_track_spots_columns = [
        'Video #', 'Video Name', 'Cell', 'Track', 'Frame', 'x', 'y', 'Intensity'
    ]
    dist_index = list(np.arange(-1*dist_range, dist_range + 1, 1))
    del dist_index[dist_range]
    dist_columns = np.array(dist_index).astype(str)
    for i in range(len(dist_columns)):
        dist_columns[i] = 'Dist ' + dist_columns[i]
    dist_columns = list(dist_columns)
    final_list_track_spots_columns += dist_columns

    output_path = csv_path + '\\_ColBD_LIFE'
    try:
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    except:
        os.mkdir(output_path)
    logging_setup(output_path, 'track-sorting')

    masks = natsorted(get_file_names_with_ext(mask_path, 'png'))
    csv_sorted = csv_name_sort_suffix(csv_path, 'spotsAll')
    csv_keys = natsorted(list(csv_sorted.keys()))

    if not len(csv_sorted.keys()) == len(masks):
        raise ValueError('Different number of Masks and Videos')

    for i in range(len(masks)):
        print_log('Processing:', masks[i])
        mask = np.swapaxes(imgio.imread(masks[i]), 0, 1)
        n_cell = np.max(mask)
        spots_video = index_format(natsorted(csv_sorted[csv_keys[i]]), n_cell)

        print_log('\t# Cells in Video:', len(spots_video))
        for j in range(n_cell):
            print_log('\t-> Cell', j, end='')
            spots_cell, _ = parse_csv_by_mask(mask, spots_video[j], j + 1)

            if _ == None:
                print_log(' [ NO LIFETIME TRACK ]')
                continue
            print_log(' [ ELIMINATED BY CELL MASK:', _, ']')

            tracks_cell = track_separation(spots_cell)
            print_log('\t\t: # Tracks in Cell:', len(tracks_cell))

            _ = 0
            for k in range(len(tracks_cell)):
                tracks_cell[k], __ = eliminate_repeated_frames(tracks_cell[k])
                _ += __
            print_log('\t\t: # Repeated Spots eliminated:', _)

            tracks_ind = []
            _ = 0
            _1 = 0
            for track in tracks_cell:
                track_ind, __, __1 = track_splitting_filtered(track, allowed_gap_max, allowed_track_length_min)
                tracks_ind += track_ind
                _ += __
                _1 += __1
            print_log('\t\t: # Splitting:', _, '# Filtered:', _1)
            print_log('\t\t: # Continuous, Individual Tracks:', len(tracks_ind))

            info = [i + 1, csv_keys[i], j + 1]
            for k in range(len(tracks_ind)):
                tracks_ind[k] = track_distance_tabulate(tracks_ind[k], dist_index, dist_none)
                for spot in tracks_ind[k]:
                    entry = info + [k + 1] + spot
                    final_list_track_spots.append(entry)

    # Output
    print_log('Saving to csv:', output_path + '\\ColBD_LIFE_tracks.csv')
    final_list_track_spots = pd.DataFrame(final_list_track_spots, columns=final_list_track_spots_columns)
    final_list_track_spots.to_csv(output_path + '\\_ColBD_LIFE_tracks.csv')
    return

'''
================================================================================================================
TRACKS
================================================================================================================
'''

def distance(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))

# Distance Calculations appended to the end of spots
def track_distance_tabulate(track, indices, dist_none):
    for i in range(len(track)):
        for j in range(len(indices)):
            if i + indices[j] < 0 or i + indices[j] >= len(track):
                track[i].append(dist_none)
            else:
                track[i].append(distance(
                    track[i][1:3], track[i + indices[j]][1:3]
                ))
    return track

# Split tracks based on gaps and filter based on length
def track_splitting_filtered(track, gap_max, len_min):
    res = []
    count_split = 0
    count_filter = 0
    record = []

    # Split
    for i in range(len(track) - 1):
        record.append(track[i])
        if track[i+1][0] - track[i][0] > gap_max:
            res.append(record.copy())
            count_split += 1
            record = []
    record.append(track[len(track) - 1])
    res.append(record.copy())

    # Filter
    i = 0
    while i < len(res):
        if(len(res[i]) < len_min):
            del res[i]
            count_filter += 1
        else:
            i += 1

    return res, count_split, count_filter

# Eliminate repeat spots in the same frame, sorted tracks
def eliminate_repeated_frames(track):
    count = 0
    res = []
    repeats = []
    scan = 0
    while scan < len(track) - 1:
        frame = track[scan][0]
        if not track[scan + 1][0] == frame:
            if len(repeats) > 0:
                repeats.append(track[scan])
                res.append(decide_spots_elimination(repeats, res[-1] if len(res) > 0 else None, track[scan + 1]))
                count += len(repeats) - 1
                repeats = []
            else:
                res.append(track[scan])
            scan += 1
        else:
            repeats.append(track[scan])
            scan += 1
    if len(repeats) > 0:
        repeats.append(track[scan])
        res.append(decide_spots_elimination(repeats, res[-1] if len(res) > 0 else None, None))
        count += len(repeats) - 1
    return res, count


def decide_spots_elimination(repeats, prev, nxt):
    best_index = -1
    best_dist = float('inf')
    for i in range(len(repeats)):
        if prev == None:
            dist = distance(repeats[i][1:3], nxt[1:3])
        elif nxt == None:
            dist = distance(repeats[i][1:3], prev[1:3])
        else:
            dist = np.mean([
                distance(repeats[i][1:3], prev[1:3]),
                distance(repeats[i][1:3], nxt[1:3])
            ])
        if dist < best_dist:
            best_index = i
            best_dist = dist
    return repeats[best_index]

# Separate each track and sort by frame number
def track_separation(spots):
    n_tracks = int(np.max(np.array(spots)[:, 0]) + 1)
    tracks = []
    for i in range(n_tracks): tracks.append([])
    for i in range(len(spots)):
        tracks[spots[i][0]].append(spots[i])
    i = 0
    while i < len(tracks):
        if len(tracks[i]) == 0:
            del tracks[i]
        else:
            i += 1
    for i in range(len(tracks)):
        for j in range(len(tracks[i])):
            tracks[i][j] = tracks[i][j][1:]
    for i in range(len(tracks)):
        tracks[i].sort(key=lambda x: x[0])
    return tracks

'''
================================================================================================================
MASKS
================================================================================================================
'''

# Read CSV and compare each spot to the mask, eliminate if outside specified cell
def parse_csv_by_mask(mask, csv, index):
    if csv == None:
        return None, None
    data = np.loadtxt(csv, delimiter=',', dtype=float)

    if data.ndim == 1:
        data = np.array([data])

    res = []

    for i in range(data.shape[0]):
        x = int(round(data[i][2]))
        y = int(round(data[i][3]))
        try:
            cell = mask[x, y]
            if (cell == index): res.append([
                int(data[i][0]), int(data[i][1]), float(data[i][2]), float(data[i][3]), int(data[i][4])
            ])
        except ValueError:
            continue
    return res, data.shape[0] - len(res)

def index_format(files, max):
    res = [None]*max
    for file in files:
        index = index_find(file)
        try:
            if(not index == -1):
                res[index - 1] = file
        except:
            raise ValueError('video mask mismatch')
    return res

def index_find(name):
    info = name.split('_')
    try:
        i = info.index('Cell')
    except ValueError: return -1
    if(i + 1 < len(info)):
        return int(info[i+1])
    else: return -1

'''
================================================================================================================
FILE HANDLING
================================================================================================================
'''

# Only keep spotsAll
def csv_name_sort_suffix(path: str, suffix:str) -> dict:
    flist = get_file_names_with_ext(path, 'csv')
    csv_sorted = {}
    for file in flist:
        fname = file.split('\\')[-1].split('_')
        if len(fname) < 4:
            continue
        if 'Cell' not in fname:
            continue
        ind = fname.index('Cell')
        video = str('_').join(fname[:ind])
        if (not video in csv_sorted):
            csv_sorted[video] = []
        if suffix in fname[ind + 2]:
            csv_sorted[video].append(file)
    return csv_sorted

def get_file_names_with_ext(path: str, ext: str):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fname = file.split('.')
            if (fname[-1] == ext):
                flist.append(root + '\\' + file)
    return flist


'''
================================================================================================================
START
================================================================================================================
'''


# Setup Logging
def logging_setup(path:str, script_name:str):
    log_file = path + '\\_ColBD_LIFE_LOG_' + script_name + '.txt'
    log_targets = [logging.FileHandler(log_file)]
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=log_targets)
    logging.StreamHandler.terminator = ''
    open(log_file, 'w').close()
    os.system('cls')


# Modified print
def print_log(*args, end='\n'):
    print(' '.join([str(a) for a in args]), end=end)
    logging.info(' '.join([str(a) for a in args] + [end]))


# Start Script
if __name__ == '__main__':
    start_time = time.time()
    main()
    print_log("--- %s seconds ---" % (time.time() - start_time))
