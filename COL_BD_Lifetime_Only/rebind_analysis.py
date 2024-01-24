import numpy as np
import pandas as pd
import time
import os
import logging
import csv
import shutil


def main():
    csv_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\Tracking'  # csv from trackmate

    # Some parameters
    rebind_distance = 2.0 # Determines same/diff particles
    min_time_bound_strict = 2
    min_time_bound_constricted = 2
    min_time_rebinding_relaxed = 2
    min_time_rebinding_strict = 2

    output_path = csv_path + '\\_ColBD_LIFE'
    log_file = output_path + '\\_ColBD_LIFE_LOG_rebind.txt'
    log_result = output_path + '\\_ColBD_LIFE_RESULT_rebind.txt'
    logging_setup(output_path, 'rebind')
    if not os.path.isdir(output_path):
        raise ValueError('Directory do not exist, please run track-sorting.py first.')

    tracks = pd.read_csv(output_path + '\\_ColBD_LIFE_bound_decisions.csv')
    tracks = tracks.loc[:, ~tracks.columns.str.contains('^Unnamed')]

    headers = tracks[['Video #', 'Cell', 'Track']].to_numpy()
    tracks = slice_tracks(tracks, headers)
    headers = np.unique(headers, axis=0)

    rebind_relaxed = []
    rebind_relaxed_spots_same = []
    rebind_relaxed_spots_diff = []
    rebind_relaxed_unsuccessful = 0
    bound_constricted = []

    rebind_strict = []
    rebind_strict_spots_same = []
    rebind_strict_spots_diff = []
    rebind_strict_unsuccessful = 0
    bound_strict = []

    for i in range(len(tracks)):
        header = headers[i]
        track = list(tracks[i][['Frame', 'x', 'y', 'Bound']].to_numpy())

        # Relaxed
        rb, rb_us, rb_same, rb_diff = rebind_record_proximity(track, rebind_distance, lambda x: not x < 1, min_time_rebinding_relaxed)
        bd = bound_record(track, lambda x: x == 1, min_time_bound_constricted)
        if(len(rb) > 0):
            rebind_relaxed += rb
        rebind_relaxed_unsuccessful += rb_us
        if(len(rb_same) > 0):
            rebind_relaxed_spots_same.append(rb_same)
        if(len(rb_diff) > 0):
            rebind_relaxed_spots_diff.append(rb_diff)
        if(len(bd) > 0):
            bound_constricted += bd

        # Strict
        rb, rb_us, rb_same, rb_diff = rebind_record_proximity(track, rebind_distance, lambda x: not x < 2, min_time_rebinding_strict)
        bd = bound_record(track, lambda x: x == 2, min_time_bound_strict)
        if(len(rb) > 0):
            rebind_strict += rb
        rebind_strict_unsuccessful += rb_us
        if(len(rb_same) > 0):
            rebind_strict_spots_same.append(rb_same)
        if(len(rb_diff) > 0):
            rebind_strict_spots_diff.append(rb_diff)
        if(len(bd) > 0):
            bound_strict += bd
        print_log('Tabulate:', 'Video', header[0], 'Cell', header[1], 'Track', header[2])

    print_log('[Analysis]')
    print_log('__________Bound__________')
    print_log('Constricted Diffusion Time (Frame):')
    print_log('->', str(pd.Series(bound_constricted).describe()).replace('\n','\n-> '),'\n')
    print_log('Strict Bound Time (Frame):')
    print_log('->', str(pd.Series(bound_strict).describe()).replace('\n','\n-> '), '\n')

    print_log('__________Rebind_________')
    print_log('Relaxed to Relaxed Rebind Probability:')
    print_log('-> Successful:', len(rebind_relaxed))
    print_log('-> Unsuccessful:', rebind_relaxed_unsuccessful)
    print_log('-> Probability', float(len(rebind_relaxed)) / float(len(rebind_relaxed) + rebind_relaxed_unsuccessful))

    print_log('\n Relaxed to Relaxed Rebind Time (Frame):')
    print_log('->', str(pd.Series([x[2] for x in rebind_relaxed]).describe()).replace('\n', '\n-> '), '\n')

    print_log('Strict to Strict Rebind Probability:')
    print_log('-> Successful:', len(rebind_strict))
    print_log('-> Unsuccessful:', rebind_strict_unsuccessful)
    print_log('-> Probability', float(len(rebind_strict)) / float(len(rebind_strict) + rebind_strict_unsuccessful))

    print_log('\n Strict to Strict Rebind Time (Frame):')
    print_log('->', str(pd.Series([x[2] for x in rebind_strict]).describe()).replace('\n', '\n-> '), '\n')

    # output, truncate log_RESULT
    with open(log_file) as fin, open(log_result, 'w') as fout:
        active = False
        for line in fin:
            if '[Analysis]' in line:
                active = True
            if active:
                fout.write(line)

    # outputs
    rebind_relaxed_spots_all = event_format_trackmate(rebind_relaxed_spots_same + rebind_relaxed_spots_diff)
    rebind_relaxed_spots_same = event_format_trackmate(rebind_relaxed_spots_same)
    rebind_relaxed_spots_diff = event_format_trackmate(rebind_relaxed_spots_diff)
    rebind_strict_spots_all = event_format_trackmate(rebind_strict_spots_diff + rebind_strict_spots_same)
    rebind_strict_spots_same = event_format_trackmate(rebind_strict_spots_same)
    rebind_strict_spots_diff = event_format_trackmate(rebind_strict_spots_diff)

    smaug_path = output_path + '\\SMAUG_REBINDING_SPOTS'
    try:
        shutil.rmtree(smaug_path)
        os.mkdir(smaug_path)
    except:
        os.mkdir(smaug_path)
    csv_write(smaug_path + '\\relaxed_rebinds_spotsAll.csv', rebind_relaxed_spots_all)
    csv_write(smaug_path + '\\relaxed_rebinds_spotsSame.csv', rebind_relaxed_spots_same)
    csv_write(smaug_path + '\\relaxed_rebinds_spotsDiff.csv', rebind_relaxed_spots_diff)
    csv_write(smaug_path + '\\strict_rebinds_spotsAll.csv', rebind_strict_spots_all)
    csv_write(smaug_path + '\\strict_rebinds_spotsSame.csv', rebind_strict_spots_same)
    csv_write(smaug_path + '\\strict_rebinds_spotsDiff.csv', rebind_strict_spots_diff)

    rebind_columns = ['From', 'To', 'Time', 'Speed', 'Distance', 'x1', 'y1', 'x2', 'y2']
    rebind_relaxed = pd.DataFrame(rebind_relaxed, columns=rebind_columns).astype({'Time': 'int'})
    rebind_strict = pd.DataFrame(rebind_strict, columns=rebind_columns).astype({'Time': 'int'})
    rebind_relaxed.to_csv(output_path + '\\_ColBD_LIFE_rebind_relaxed.csv')
    rebind_strict.to_csv(output_path + '\\_ColBD_LIFE_rebind_strict.csv')
    return

def event_format_trackmate(events):
    formatted = []
    i = 1
    for track in events:
        for event in track:
            for spot in event:
                formatted.append([i, spot[0], spot[1], spot[2], 10000])
            i += 1
    return formatted

def bound_record(track, criteria, min_time):
    bound = []
    record = track[0][3]
    event = []
    f = 0
    while f < len(track):
        if record == track[f][3]:
            if criteria(track[f][3]):
                event.append(track[f])
        else:
            if(len(event) > 0):
                time = 1 if len(event) == 1 else int(event[-1][0] - event[0][0] + 1)
                if(time < min_time):
                    event = []
                else:
                    bound.append(event.copy())
                    event = []
            if(criteria(track[f][3])):
                event.append(track[f])
            record = track[f][3]
        f += 1
    if(len(event) > 0):
        time = 1 if len(event) == 1 else int(event[-1][0] - event[0][0] + 1)
        if (time < min_time):
            event = []
        else: bound.append(event.copy())

    result = []
    for event in bound:
        if(len(event) == 1):
            result.append(1)
        else:
            result.append(int(event[-1][0] - event[0][0] + 1))
    return result

def rebind_record_proximity(track, rebind_distance, criteria, min_time):
    rebinds = []
    event = []
    events_same = []
    events_diff = []
    active = False
    record_pos = [track[0][1], track[0][2]]
    f = 0

    while (f < len(track)):
        if (len(event) > 0 and criteria(track[f][3])):
            pos = [track[f][1], track[f][2]]
            dist = distance(pos, record_pos)
            table = rebind_tabulate(event.copy(), 0, 0) # just to get the time
            if(table[2] < min_time): # min_time threshold
                event = []
            else:
                if (dist > rebind_distance):
                    prev, nxt = 1, 2
                    events_diff.append(event.copy())
                else:
                    prev, nxt = 1, 1
                    events_same.append(event.copy())
                rebinds.append(
                    table + [dist, record_pos[0], record_pos[1], pos[0], pos[1]])
                event = []
        if criteria(track[f][3]):
            active = True
            record_pos = [track[f][1], track[f][2]]
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

def csv_write(path, data):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for line in data:
            writer.writerow(line)
        file.close()

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
