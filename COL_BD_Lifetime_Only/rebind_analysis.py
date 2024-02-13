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
    rebind_distance_same = configs['rebind-analysis']['rebind_distance_same']
    rebind_distance_diff = configs['rebind-analysis']['rebind_distance_diff']
    min_time_bound_strict = configs['rebind-analysis']['min_time_bound_strict']
    min_time_bound_constricted = configs['rebind-analysis']['min_time_bound_constricted']
    min_time_rebinding_relaxed = configs['rebind-analysis']['min_time_rebinding_relaxed']
    min_time_rebinding_strict = configs['rebind-analysis']['min_time_rebinding_strict']
    min_time_diffusion = configs['rebind-analysis']['min_time_diffusion']
    min_time_diffusion_subsequent = configs['rebind-analysis']['min_time_diffusion_subsequent']
    max_time_rebinding = configs['rebind-analysis']['max_time_rebinding']
    max_time_constrained = configs['rebind-analysis']['max_time_constrained']

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
    rebind_relaxed_spots_all = []
    rebind_relaxed_unsuccessful = 0
    bound_constricted = []
    bound_constricted_record = []

    rebind_relaxed_spots_entiretrack = []

    rebind_strict = []
    rebind_strict_spots_same = []
    rebind_strict_spots_diff = []
    rebind_strict_spots_all = []
    rebind_strict_unsuccessful = 0
    bound_strict = []
    bound_strict_record = []

    rebind_strict_spots_entiretrack = []

    constrained_dest = np.array([0, 0])

    fast_diffusion_dest = np.array([0, 0])
    fast_diffusion_dest_strict = np.array([0, 0])
    fast_diffusion_time = []

    all_diffusion_dest = np.array([0, 0])
    all_diffusion_dest_strict = np.array([0, 0])
    all_diffusion_time = []

    proportion_count = np.array([0, 0, 0])

    for i in range(len(tracks)):
        header = headers[i]
        track = list(tracks[i][['Frame', 'x', 'y', 'Bound']].to_numpy())

        # Relaxed
        rb, rb_us, rb_same, rb_diff, rb_all = (
            rebind_record_proximity(track, rebind_distance_same, rebind_distance_diff, lambda x: not x < 1, min_time_rebinding_relaxed, max_time_rebinding))
        bd = bound_record(track, lambda x: x == 1, min_time_bound_constricted)
        if(len(rb) > 0):
            for j in range(len(rb)):
                rb[j] = list(header) + rb[j]
            rebind_relaxed += rb
            rebind_relaxed_spots_entiretrack.append([track.copy()])
        rebind_relaxed_unsuccessful += rb_us
        if(len(rb_same) > 0):
            rebind_relaxed_spots_same.append(rb_same)
        if(len(rb_diff) > 0):
            rebind_relaxed_spots_diff.append(rb_diff)
        if(len(rb_all) > 0):
            rebind_relaxed_spots_all.append(rb_all)
        if(len(bd) > 0):
            bound_constricted += bd
        j = 1
        for bdframe in bd:
            bound_constricted_record.append(list(header.copy()) + [j, bdframe])
            j += 1

        # Strict
        rb, rb_us, rb_same, rb_diff, rb_all = (
            rebind_record_proximity(track, rebind_distance_same, rebind_distance_diff, lambda x: not x < 2, min_time_rebinding_strict, max_time_rebinding))
        bd = bound_record(track, lambda x: x == 2, min_time_bound_strict)
        if(len(rb) > 0):
            for j in range(len(rb)):
                rb[j] = list(header) + rb[j]
            rebind_strict += rb
            rebind_strict_spots_entiretrack.append([track.copy()])
        rebind_strict_unsuccessful += rb_us
        if(len(rb_same) > 0):
            rebind_strict_spots_same.append(rb_same)
        if(len(rb_diff) > 0):
            rebind_strict_spots_diff.append(rb_diff)
        if(len(rb_all) > 0):
            rebind_strict_spots_all.append(rb_all)
        if(len(bd) > 0):
            bound_strict += bd
        j = 1
        for bdframe in bd:
            bound_strict_record.append(list(header.copy()) + [j, bdframe])
            j += 1

        # constrained diffusion
        constrained_dest = np.add(constrained_dest, constrained_record(track, min_time_bound_constricted, min_time_bound_strict, max_time_constrained))

        # fast diffusion
        df_time, df_counts = diffusion_record(track, lambda x: x > 0, min_time_diffusion, min_time_diffusion_subsequent)
        fast_diffusion_time += df_time
        fast_diffusion_dest = np.add(fast_diffusion_dest, df_counts)
        if len(bd) > 0:
            fast_diffusion_dest_strict = np.add(fast_diffusion_dest_strict, df_counts)

        # all diffusion
        df_time, df_counts = diffusion_record(track, lambda x: x > 1, min_time_diffusion,
                                              min_time_diffusion_subsequent)
        all_diffusion_time += df_time
        all_diffusion_dest = np.add(all_diffusion_dest, df_counts)
        if len(bd) > 0:
            all_diffusion_dest_strict = np.add(all_diffusion_dest_strict, df_counts)

        # proportion count
        p_counts = label_count(track)
        proportion_count = np.add(proportion_count, p_counts)

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
    print_log('->', str(pd.Series([x[5] for x in rebind_relaxed]).describe()).replace('\n', '\n-> '), '\n')

    print_log('Strict to Strict Rebind Probability:')
    print_log('-> Successful:', len(rebind_strict))
    print_log('-> Unsuccessful:', rebind_strict_unsuccessful)
    print_log('-> Probability', float(len(rebind_strict)) / float(len(rebind_strict) + rebind_strict_unsuccessful))

    print_log('\n Strict to Strict Rebind Time (Frame):')
    print_log('->', str(pd.Series([x[5] for x in rebind_strict]).describe()).replace('\n', '\n-> '), '\n')

    print_log('__________Constrained_____')
    print_log('Count of Constrained to Diffusion:', constrained_dest[0])
    print_log('Count of Constrained to Bound:', constrained_dest[1])
    print_log('Probability of Constrained to Bound:', float(constrained_dest[1]) / float(constrained_dest[0] + constrained_dest[1]))
    print_log('')

    print_log('______Diffusion_Fast______')
    print_log('Fast Diffusion all tracks by Frame')
    print_log('-> Count of Fast Diffusion to Fast Diffusion:', fast_diffusion_dest[0])
    print_log('-> Count of Fast Diffusion to Constrict/Bound:', fast_diffusion_dest[1])
    print_log('-> Probability of Fast Diffusion to Constrict/Bound:', float(fast_diffusion_dest[1]) / float(fast_diffusion_dest[0] + fast_diffusion_dest[1]))

    print_log('\nFast Diffusion tracks with strict binding by Frame')
    print_log('-> Count of Fast Diffusion to Fast Diffusion:', fast_diffusion_dest_strict[0])
    print_log('-> Count of Fast Diffusion to Constrict/Bound:', fast_diffusion_dest_strict[1])
    print_log('-> Probability of Fast Diffusion to Constrict/Bound:', float(fast_diffusion_dest_strict[1]) / float(fast_diffusion_dest_strict[0] + fast_diffusion_dest_strict[1]))

    print_log('\nFast Diffusion average time (Frame):')
    print_log('->', str(pd.Series(fast_diffusion_time).describe()).replace('\n', '\n-> '), '\n')

    print_log('______Diffusion_All_______')
    print_log('All Diffusion all tracks by Frame')
    print_log('-> Count of Diffusion to Diffusion:', all_diffusion_dest[0])
    print_log('-> Count of Diffusion to Bound:', all_diffusion_dest[1])
    print_log('-> Probability of Diffusion to Bound:',
              float(all_diffusion_dest[1]) / float(all_diffusion_dest[0] + all_diffusion_dest[1]))

    print_log('\nAll Diffusion tracks with strict binding by Frame')
    print_log('-> Count of Diffusion to Diffusion:', all_diffusion_dest_strict[0])
    print_log('-> Count of Diffusion to Bound:', all_diffusion_dest_strict[1])
    print_log('-> Probability of Diffusion to Bound:',
              float(all_diffusion_dest_strict[1]) / float(all_diffusion_dest_strict[0] + all_diffusion_dest_strict[1]))

    print_log('\nAll Diffusion average time (Frame):')
    print_log('->', str(pd.Series(all_diffusion_time).describe()).replace('\n', '\n-> '), '\n')

    print_log('____Counted_Proportions___')
    print_log('Count of all frames:', np.sum(proportion_count))
    print_log('Count of fast diffusion: ', proportion_count[0], '-> Proportion:',
              float(proportion_count[0]) / np.sum(proportion_count))
    print_log('Count of constrained diffusion: ', proportion_count[1], '-> Proportion:',
              float(proportion_count[1]) / np.sum(proportion_count))
    print_log('Count of strict binding: ', proportion_count[2], '-> Proportion:',
              float(proportion_count[2]) / np.sum(proportion_count))


    # output, truncate log_RESULT
    with open(log_file) as fin, open(log_result, 'w') as fout:
        active = False
        for line in fin:
            if '[Analysis]' in line:
                active = True
            if active:
                fout.write(line)

    # outputs
    rebind_relaxed_spots_all = event_format_trackmate(rebind_relaxed_spots_all)
    rebind_relaxed_spots_same = event_format_trackmate(rebind_relaxed_spots_same)
    rebind_relaxed_spots_diff = event_format_trackmate(rebind_relaxed_spots_diff)
    rebind_relaxed_spots_entiretrack = event_format_trackmate(rebind_relaxed_spots_entiretrack)
    rebind_strict_spots_all = event_format_trackmate(rebind_strict_spots_all)
    rebind_strict_spots_same = event_format_trackmate(rebind_strict_spots_same)
    rebind_strict_spots_diff = event_format_trackmate(rebind_strict_spots_diff)
    rebind_strict_spots_entiretrack = event_format_trackmate(rebind_strict_spots_entiretrack)

    smaug_path = output_path + '\\SMAUG_REBINDING_SPOTS'
    try:
        shutil.rmtree(smaug_path)
        os.mkdir(smaug_path)
    except:
        os.mkdir(smaug_path)
    csv_write(smaug_path + '\\relaxed_rebinds_spotsAll.csv', rebind_relaxed_spots_all)
    csv_write(smaug_path + '\\relaxed_rebinds_spotsSame.csv', rebind_relaxed_spots_same)
    csv_write(smaug_path + '\\relaxed_rebinds_spotsDiff.csv', rebind_relaxed_spots_diff)
    csv_write(smaug_path + '\\relaxed_rebinds_spotsTrack.csv', rebind_relaxed_spots_entiretrack)
    csv_write(smaug_path + '\\strict_rebinds_spotsAll.csv', rebind_strict_spots_all)
    csv_write(smaug_path + '\\strict_rebinds_spotsSame.csv', rebind_strict_spots_same)
    csv_write(smaug_path + '\\strict_rebinds_spotsDiff.csv', rebind_strict_spots_diff)
    csv_write(smaug_path + '\\strict_rebinds_spotsTrack.csv', rebind_strict_spots_entiretrack)

    rebind_columns = ['Video #', 'Cell', 'Track', 'From', 'To', 'Time', 'Speed', 'Distance', 'x1', 'y1', 'x2', 'y2']
    rebind_relaxed = pd.DataFrame(rebind_relaxed, columns=rebind_columns).astype({'Time': 'int'})
    rebind_strict = pd.DataFrame(rebind_strict, columns=rebind_columns).astype({'Time': 'int'})
    rebind_relaxed.to_csv(output_path + '\\_ColBD_LIFE_rebind-relaxed.csv')
    rebind_strict.to_csv(output_path + '\\_ColBD_LIFE_rebind-strict.csv')

    boundtime_columns = ['Video #', 'Cell', 'Track', 'Event', 'Bound Time']
    bound_constricted_record = pd.DataFrame(bound_constricted_record, columns=boundtime_columns).astype({'Bound Time': 'int'})
    bound_strict_record = pd.DataFrame(bound_strict_record, columns=boundtime_columns).astype({'Bound Time': 'int'})
    bound_constricted_record.to_csv(output_path + '\\_ColBD_LIFE_bound-constricted.csv')
    bound_strict_record.to_csv(output_path + '\\_ColBD_LIFE_bound-strict.csv')
    return

def label_count(track):
    result = np.array([0, 0, 0])
    labels = np.array(track)[:, 3]
    unique, counts = np.unique(labels, return_counts=True)
    unique = unique.astype('int')
    for i in range(unique.shape[0]):
        result[unique[i]] = counts[i]
    return result

def event_format_trackmate(events):
    formatted = []
    i = 1
    for track in events:
        for event in track:
            for spot in event:
                formatted.append([i, spot[0], spot[1], spot[2], 10000])
            i += 1
    return formatted

def diffusion_record(track, criteria, min_time, min_time_bound):
    counts = [0, 0]
    event = []
    event2 = []
    diffusion_time = []
    f = 0
    while f < len(track):
        if not criteria(track[f][3]):
            event.append(track[f])
        else:
            if len(event) > 0:
                time_int = 1 if len(event) == 1 else event[-1][0] - event[0][0] + 1
                if time_int < min_time:
                    event = []
                    continue
                diffusion_time.append(time_int)
                counts[0] += len(event) - 1
                record = track[f][3]
                i = f
                while i < len(track) and criteria(track[i][3]):
                    event2.append(track[i])
                    i += 1
                time_int2 = 1 if len(event2) == 1 else event2[-1][0] - event2[0][0] + 1
                if time_int2 < min_time_bound:
                    counts[0] += 1
                else:
                    counts[1] += 1
                event2 = []
                event = []
        f += 1
    if len(event) > 0:
        time_int = 1 if len(event) == 1 else event[-1][0] - event[0][0] + 1
        if time_int >= min_time:
            diffusion_time.append(time_int)
            counts[0] += len(event) - 1
    return diffusion_time, np.array(counts)

def constrained_record(track, min_time_constricted, min_time_bound, max_time_constrained):
    counts = [0, 0]
    f = 0
    event = []
    event2 = []
    while f < len(track):
        if track[f][3] == 1:
            event.append(track[f])
        else:
            if(len(event) > 0):
                time_int = 1 if len(event) == 1 else event[-1][0] - event[0][0] + 1
                record = track[f][3]
                i = f
                while i < len(track) and track[i][3] == record:
                    event2.append(track[i])
                    i += 1
                time_int2 = 1 if len(event2) == 1 else event2[-1][0] - event2[0][0] + 1
                if(time_int >= min_time_constricted and time_int <= max_time_constrained and
                        time_int2 >= min_time_bound):
                    counts[0 if track[f][3] == 0 else 1] += 1
                event2 = []
                event = []
        f += 1

    return np.array(counts)

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

def rebind_record_proximity(track, rebind_distance_same, rebind_distance_diff, criteria, min_time, max_time):
    rebinds = []
    event = []
    events_same = []
    events_diff = []
    events_all = []
    active = False
    record_pos = [track[0][1], track[0][2]]
    record_f = 0
    unsuccessful = 0
    f = 0

    while (f < len(track)):
        if (len(event) > 0 and criteria(track[f][3])):
            pos = [track[f][1], track[f][2]]
            dist = distance(pos, record_pos)
            table = rebind_tabulate(event.copy(), 0, 0) # just to get the time
            if(table[2] < min_time): # min_time threshold
                event = []
            elif(table[2] > max_time):
                event = []
                unsuccessful += 1
            else:
                if (dist >= rebind_distance_diff):
                    prev, nxt = 1, 2
                    events_diff.append(event.copy())
                elif (dist <= rebind_distance_same):
                    prev, nxt = 1, 1
                    events_same.append(event.copy())
                else:
                    prev, nxt = 1, 2
                events_all.append(event.copy())
                time_int = rebind_trace_avg(track, f, criteria, 1)[0]
                rebinds.append(
                    rebind_tabulate(event.copy(), prev, nxt) + [dist] +
                    rebind_trace_avg(track, record_f - 1, criteria, -1)[1] +
                    rebind_trace_avg(track, f, criteria, 1)[1]
                )
                event = []
        if criteria(track[f][3]):
            active = True
            record_pos = [track[f][1], track[f][2]]
        elif (active):
            if len(event) == 0:
                record_f = f
            event.append(track[f])
        f += 1

    # unsuccessful event
    unsuccessful += 1 if (len(event) > 0) else 0
    return rebinds, unsuccessful, events_same, events_diff, events_all


def rebind_trace_avg(track, sframe, criteria, dir):
    f = sframe
    x = []
    y = []
    event = []
    while f >= 0 and f < len(track) and criteria(track[f][3]):
        event.append(track[f])
        x.append(track[f][1])
        y.append(track[f][2])
        f += dir
    time_int = 1 if len(event) == 1 else int(event[-1][0] - event[0][0] + 1)
    return time_int, [np.mean(x), np.mean(y)]


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
