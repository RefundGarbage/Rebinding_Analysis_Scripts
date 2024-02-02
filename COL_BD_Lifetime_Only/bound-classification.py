import numpy as np
import pandas as pd
import time
import os
import logging
from tqdm import trange

def determine_bound(before:list, after:list):
    if before[0] == 1 and after[0] ==1:
        return 1
    elif before[0] == 1 or after[0] == 1:
        return 1 if np.average(before[1:4]) > 0.5 or np.average(after[1:4]) > 0.5 else 0
    else: return 0

def determine_bound_strict(before:list, after:list, before_1:list, after_1:list):
    decision = np.max([
        float(np.average(before[0:4])), float(np.average(after[0:4])),
        1.0 if np.average(np.concatenate([before[0:3], after[0:3]])) > 0.8 else 0.0,
        1.0 if np.average(np.concatenate([before[0:1], after[0:2]])) == 1.0 else 0.0,
        1.0 if np.average(np.concatenate([before[0:2], after[0:1]])) == 1.0 else 0.0,
        
    ])
    return 1 if decision >= 1 else 0

def main():

    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\SSB113\\5\\timelapse'
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\AnalysisRebindCBC_start0_Quality5'  #csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA7\\timelapse\\AnalysisRebindCBC_start0_Quality5'
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA3\\Timelapse\\AnalysisRebindCBC_start0_Quality5'  # csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\SSB113\\timelapse'
    csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA5\\timelapse'

    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\set3' 
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA3\\Timelapse\\set2'  # csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA7\\timelapse\\set2' 
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\AnalysisRebindCBC_start0_Quality5'

    #csv_path = 'F:\\_Microscopy\\Rawdates\\_RUNreb1108_4diameter\\timelapse\\GOOD_OLD_Quality2point5\\pr208\\pr208_AnalysisRebindCBC1114_6diam_Dog'
    #csv_path = 'F:\\_Microscopy\\Rawdates\\_RUNreb1108_4diameter\\timelapse\\GOOD_OLD_Quality2point5\\pr208n\\AnalysisRebindCBC_11146diam_DOG'  # csv from trackmate

    #csv_path = 'F:\\_Microscopy\\Rawdates\\_RUNreb1108_4diameter\\timelapse\\GOOD_OLD_Quality2point5\\pr212\\AnalysisRebindCBC_11146dia_dog'
    #csv_path = 'F:\\_Microscopy\\Rawdates\\20230913_ypetB_haloQ\\Images\\timelapse\\101023\\AnalysisRebindCBC_1010123_start0'  # csv from trackmate
    #csv_path = 'F:\\_Microscopy\\Rawdates\\202301109_TetR_haloQmutnWT\\Images\\COMB\\AnalysisRebindCBC_5quality\\PR208' # csv from trackmate
    #csv_path = 'F:\\_Microscopy\\Rawdates\\202301109_TetR_haloQmutnWT\\Images\\COMB\\AnalysisRebindCBC_5quality\\PR212' # csv from trackmate
    #csv_path = 'F:\\_Microscopy\\Rawdates\\202309026_pr208DmHQctd\\Images\\AnalysisRebindCBCstart0'  # csv from trackmate
    #csv_path = 'F:\\_Microscopy\\Rawdates\\_RUNreb1108_4diameter\\timelapse\\GOOD_OLD_Quality2point5\\pr212\\AnalysisRebindCBC_11146dia_dog'  # csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\SSB113\\timelapse'  # csv from trackmate


    # Some parameters
    distance_threshold = 3.0
    distance_threshold_strict = 1.6

    output_path = csv_path + '\\_ColBD_LIFE'
    if not os.path.isdir(output_path):
        raise ValueError('Directory do not exist, please run track-sorting.py first.')
    logging_setup(output_path, 'bound-classification')

    print_log('Reading from csv:', output_path + '\\_ColBD_LIFE_tracks.csv')
    tracks = pd.read_csv(output_path + '\\_ColBD_LIFE_tracks.csv')
    tracks = tracks.loc[:, ~tracks.columns.str.contains('^Unnamed')]

    headers = tracks[['Video #', 'Cell', 'Track']].to_numpy()
    tracks = slice_tracks(tracks, headers)

    track_results_threshold = []
    track_results_threshold_strict = []
    decisions_bound = []
    decisions_bound_strict = []
    decisions_bound_cdiffusion = []
    decisions_bound_overall = []
    for i in trange(len(tracks)):
        track = tracks[i]
        # Relaxed
        result_before, result_after = process_track(track, distance_threshold)
        track_results_threshold.append(result_before[result_before.columns[::-1]].join(result_after))
        decision_bound = []
        for j in range(len(result_before.index)):
            spot_before = result_before.iloc[j].to_numpy()
            spot_after = result_after.iloc[j].to_numpy()
            decision_bound.append(determine_bound(spot_before, spot_after))
        decisions_bound.append(pd.Series(decision_bound))

        # Strict
        result_before, result_after = process_track(track, distance_threshold_strict)
        track_results_threshold_strict.append(result_before[result_before.columns[::-1]].join(result_after))
        decision_bound_strict = []
        record = [np.array([]), np.array([])]
        for j in range(len(result_before.index)):
            spot_before = result_before.iloc[j].to_numpy()
            spot_after = result_after.iloc[j].to_numpy()
            decision_bound_strict.append(determine_bound_strict(spot_before, spot_after, record[0], record[1]))
            record = [spot_before.copy(), spot_after.copy()]
        decisions_bound_strict.append(pd.Series(decision_bound_strict))

        # Constricted Diffusion
        decisions_bound_cdiffusion.append(pd.Series(np.logical_xor(decision_bound, decision_bound_strict).astype('int')))

        # Overall Bound 0, 1, 2
        decisions_bound_overall.append(pd.Series(np.add(decision_bound, decision_bound_strict)))

    output_relaxed = []
    output_strict = []
    output_both = []

    for i in trange(len(tracks)):
        track = tracks[i]
        result_relaxed = track_results_threshold[i]
        decision_relaxed = decisions_bound[i]
        output_relaxed.append(
            track[track.columns.tolist()[0:5] + track.columns.tolist()[8:]].reset_index(drop=True)
            .join(result_relaxed.assign(Bound=decision_relaxed).reset_index(drop=True))
        )
        result_strict = track_results_threshold_strict[i]
        decision_strict = decisions_bound_strict[i]
        output_strict.append(
            track[track.columns.tolist()[0:5] + track.columns.tolist()[8:]].reset_index(drop=True)
            .join(result_strict.assign(Bound=decision_strict).reset_index(drop=True))
        )
        decision_cdiffusion = decisions_bound_cdiffusion[i]
        decision_overall = decisions_bound_overall[i]
        output_both.append(
            track[track.columns.tolist()[:8]].reset_index(drop=True)
            .assign(RelaxedBound=decision_relaxed, StrictBound=decision_strict)
            .assign(ContrictedDiffusion=decision_cdiffusion)
            .assign(Bound=decision_overall)
        )
    pd.concat(output_relaxed).to_csv(output_path + '\\_ColBD_LIFE_bound_relaxed.csv')
    pd.concat(output_strict).to_csv(output_path + '\\_ColBD_LIFE_bound_strict.csv')
    pd.concat(output_both).to_csv(output_path + '\\_ColBD_LIFE_bound_decisions.csv')
    return

'''
================================================================================================================
TRACKS
================================================================================================================
'''

def process_track(track, threshold:float):
    dist_columns = [c for c in track.columns if 'Dist' in c]
    dist_range = int(len(dist_columns) / 2.0)
    dist_columns_before = dist_columns[:dist_range][::-1]
    dist_columns_after = dist_columns[dist_range:]

    dist_before = track[dist_columns_before].to_numpy()
    dist_after = track[dist_columns_after].to_numpy()

    result_before_columns = ['B' + str(i+1) for i in range(dist_range)]
    result_after_columns = ['A' + str(i+1) for i in range(dist_range)]

    result_before = pd.DataFrame((dist_before < threshold).astype('int'), columns=result_before_columns)
    result_after = pd.DataFrame((dist_after < threshold).astype('int'), columns=result_after_columns)
    return result_before, result_after

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
