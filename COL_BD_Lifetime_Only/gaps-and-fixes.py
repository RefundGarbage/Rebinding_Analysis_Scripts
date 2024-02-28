import numpy as np
import pandas as pd
import time
import os
import logging
import csv
import shutil
import tomllib

def main():
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config_path = os.path.join(__location__, 'script-config.toml')
    with open(config_path, 'rb') as config_file:
        configs = tomllib.load(config_file)

    csv_path = configs['path']['csv_path']

    # Some parameters
    min_time_strict = configs['gaps-and-fixes']['min_time_strict']
    min_time_constrained = configs['gaps-and-fixes']['min_time_constrained']
    min_time_diffusion = configs['gaps-and-fixes']['min_time_diffusion']
    max_bound_gapFill = configs['gaps-and-fixes']['max_bound_gapFill']

    output_path = csv_path + '\\' + configs['path']['output_folder_name']
    if not os.path.isdir(output_path):
        raise ValueError('Directory do not exist, please run track-sorting.py first.')
    logging_setup(output_path, 'gaps-and-fixes')

    print_log('Reading from csv:', output_path + '\\_ColBD_LIFE_bound_decisions.csv')
    tracks = pd.read_csv(output_path + '\\_ColBD_LIFE_bound_decisions.csv')
    tracks = tracks.loc[:, ~tracks.columns.str.contains('^Unnamed')]

    headers = tracks[['Video #', 'Cell', 'Track']].to_numpy()
    tracks = slice_tracks(tracks, headers)

    counts_gap = 0
    counts_event = 0
    counts_operations = 0
    output_tracks = []

    for i in range(len(tracks)):
        header = headers[i]
        track = tracks[i]
        pos = track[['x', 'y']].to_numpy()
        vname = track.iloc[0]['Video Name']
        track = track.loc[:, ~track.columns.str.startswith(('Video Name', 'x', 'y'))]
        track = list(track.to_numpy())
        print_log('Fixing:', 'Video', header[0], 'Cell', header[1], 'Track', header[2])

        # Fill Gaps
        _, track, pos = process_gaps(track, pos, lambda l, r, dur: 1 if l == 2 and r == 2 and dur > max_bound_gapFill else min(l, r))
        print_log('\t-> Gap:', _, 'filled')
        counts_gap += _

        # Separate Events
        _, events = event_separation(track)
        print_log('\t-> Events:', _, 'found')
        counts_event += _

        __ = 0

        # Pass 3: CD -> FD, Merge
        _, events1 = pass_events(events, 1, lambda l,r: 2 if l == 2 and r == 2 else 0, min_time_constrained)
        print_log('\t-> Pass 3:', _, 'events relabeled.')
        __ += _

        # Pass 1: FD -> CD, Merge
        _, events2 = pass_events(events1, 0, 1, min_time_diffusion)
        print_log('\t-> Pass 1:', _, 'events relabeled.')
        __ += _

        # Pass 2: SB -> CD, Merge
        _, events3 = pass_events(events2, 2, 1, min_time_strict)
        print_log('\t-> Pass 2:', _, 'events relabeled.')
        __ += _



        print_log('\t-> Pass Complete:', __, 'relabeling performed,', len(events3), 'events left.')
        counts_operations += __

        # Collect Events -> Reconstruct Track
        track1 = events_to_track(events1)
        track2 = events_to_track(events2)
        track3 = events_to_track(events3)

        trackdf = pd.DataFrame(np.array(track3)[:, :5], columns=['Video #', 'Cell', 'Track', 'Frame', 'Intensity'])
        trackdf = (trackdf.assign(GapFixed=np.array(track)[:, 8],
                                  Pass1=np.array(track1)[:, 8], Pass2=np.array(track2)[:, 8], Pass3=np.array(track3)[:, 8],
                                 Bound=np.array(track3)[:, 8], isGap=np.array(track3)[:, 9],
                                 Name=np.array([vname]*len(track3)))
                   .join(pd.DataFrame(np.array(pos), columns=['x', 'y']))
                   .rename(columns={'Name':'Video Name'}))
        trackdf = trackdf[['Video #', 'Video Name', 'Cell', 'Track', 'Frame', 'x', 'y', 'Intensity',
                                    'isGap', 'GapFixed', 'Pass1', 'Pass2', 'Pass3', 'Bound']]
        output_tracks.append(trackdf)

    print_log('__________________________________________________')
    print_log('Complete: \n\t-> Frame Gap Filled:', counts_gap,
              '\n\t-> Events Separated:', counts_event,
              '\n\t-> Relabeling Performed:', counts_operations
              )

    print_log('Saving to:', output_path + '\\_ColBD_LIFE_gaps-and-fixes_decisions.csv')
    pd.concat(output_tracks).to_csv(output_path + '\\_ColBD_LIFE_gaps-and-fixes_decisions.csv')
    return

# Reconstruct Track from Events
def events_to_track(events):
    track = []
    for bvr, event in events:
        track += event
    return track

# Pass: relabel events with track length less than min time, then merge tracks with same behaviors
def pass_events(events, ori, sub, min_time):
    count = 0
    behaviors = [[0, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 2]]
    events = events.copy()
    for i in range(len(events)):
        bvr, event = events[i]
        event = event.copy()
        if not type(sub) == int:
            l = -1 if i-1 < 0 else events[i-1][0]
            r = -1 if i+1 >= len(events) else events[i+1][0]
        if not bvr == ori:
            continue
        if len(event) >= min_time:
            continue
        count += 1
        for f in range(len(event)):
            if type(sub) == int:
                event[f] = np.array(list(event[f][:5]) + behaviors[sub] + [event[f][9]])
            else:
                event[f] = np.array(list(event[f][:5]) + behaviors[sub(l, r)] + [event[f][9]])
        if type(sub) == int:
            events[i] = (sub, event)
        else:
            events[i] = (sub(l,r), event)
    result_events = []
    record_events = [events[0]]
    i = 1
    while i < len(events):
        if not events[i][0] == record_events[-1][0]:
            if len(record_events) == 1:
                result_events += record_events
                record_events = []
            else:
                result_events.append(merge_events(record_events))
                record_events = []
        record_events.append(events[i])
        i += 1
    if len(record_events) == 1:
        result_events += record_events
    else:
        result_events.append(merge_events(record_events))
    return count, result_events

# merge events with the same behavior
def merge_events(events:list):
    event = []
    for i in range(len(events)):
        event += events[i][1]
    return (events[0][0], event)

# event separation, list(tuples(int, list(ndarray))))
def event_separation(track):
    # [f][8] -> bound
    count = 0
    events = []
    event = [track[0]]
    f = 1
    while f < len(track):
        if track[f][8] == event[-1][8]:
            event.append(track[f])
        else:
            events.append((event[-1][8], event))
            count += 1
            event = [track[f]]
        f += 1
    if len(event) > 0:
        events.append((event[-1][8], event))
        count += 1
    return count, events


# gap fixes, by frame
def process_gaps(track, pos, criteria):
    # [f][3] -> frame #
    # [f][8] -> bound
    result = []
    result_pos = []
    behaviors = [[0, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 2]]
    count = 0
    f = 1
    result.append(np.array(list(track[0].copy()) + [0]))
    result_pos.append(pos[0])
    while f < len(track):
        if(track[f][3] - track[f-1][3] < 2):
            result.append(np.array(list(track[f].copy()) + [0]))
            result_pos.append(pos[f])
            f += 1
            continue
        fr = track[f-1][3] + 1
        bound = behaviors[criteria(track[f-1][8], track[f][8], track[f][3]-track[f-1][3] + 1)] # Behavior
        template = list(track[f-1][:3]).copy()

        # Gap filling
        while fr < track[f][3]:
            result.append(
                np.array(template.copy() + [fr, 0] + bound.copy() + [-1])
            )
            result_pos.append(np.array([-1, -1]))
            count += 1
            fr += 1
        result.append(np.array(list(track[f].copy()) + [0]))
        result_pos.append(pos[f])
        f += 1

    return count, result, result_pos


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
