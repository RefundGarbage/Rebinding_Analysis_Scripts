import numpy as np
import pandas as pd
import time
import os
from natsort import natsorted
import logging
import shutil


def main():
    csv_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\Tracking'  # csv from trackmate

    # Some parameters
    rebind_distance = 2.0
    relaxed = True

    output_path = csv_path + '\\_ColBD_LIFE'
    logging_setup(output_path, 'rebind')
    if not os.path.isdir(output_path):
        raise ValueError('Directory do not exist, please run track-sorting.py first.')

    tracks = pd.read_csv(output_path + '\\_ColBD_LIFE_bound_decisions.csv')
    tracks = tracks.loc[:, ~tracks.columns.str.contains('^Unnamed')]

    headers = tracks[['Video #', 'Cell', 'Track']].to_numpy()
    tracks = slice_tracks(tracks, headers)

    rebind = []
    rebind_spots_same = []
    rebind_spots_diff = []
    rebind_unsuccessful = 0
    for i in range(len(tracks)):
        track = list(tracks[i][['Frame', 'x', 'y', 'Bound']].to_numpy())
        _ = rebind_record_proximity(track, rebind_distance)
        print(i)
    return


def rebind_record_proximity(track, rebind_distance):
    rebinds = []
    event = []
    events_same = []
    events_diff = []
    active = False
    record = track[0][3]
    record_pos = [track[0][1], track[0][2]]
    f = 0
    i = 1

    while (f < len(track)):
        if (len(event) > 0 and not track[f][3] == 0):
            pos = [track[f][1], track[f][2]]
            dist = distance(pos, record_pos)
            if (dist > rebind_distance):
                prev, nxt = 1, 2
                events_diff.append(event.copy())
            else:
                prev, nxt = 1, 1
                events_same.append(event.copy())
            rebinds.append(
                [i] + rebind_tabulate(event.copy(), prev, nxt) + [dist, record_pos[0], record_pos[1], pos[0], pos[1]])
            event = []
            record = track[f][3]
            i += 1
        if not (track[f][3] == 0):
            active = True
            record_pos = [track[f][1], track[f][2]]
            record = track[f][3]
        elif (active):
            event.append(track[f])
        f += 1

    # unsuccessful event
    unsuccessful = 1 if (len(event) > 0) else 0
    return rebinds, unsuccessful, events_same, events_diff


def rebind_tabulate(segment, prev, nxt):
    frames = [s[0] for s in segment]
    rebinding_time = max(frames) - min(frames) + 1
    distances = [0]
    if (len(segment) > 1):
        for i in range(1, len(segment)):
            distances.append(distance([segment[i - 1][1], segment[i - 1][2]], [segment[i][1], segment[i][2]]))
    return [prev, nxt, rebinding_time, sum(distances) / rebinding_time if (len(segment) > 1) else 1]

def distance(p1, p2):
    return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1] - p2[1], 2))

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
