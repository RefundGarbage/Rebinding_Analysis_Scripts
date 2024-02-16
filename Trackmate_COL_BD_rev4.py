import numpy as np
from scipy import io as scio
from skimage import io as imgio
import os
from natsort import natsorted
import time
import csv
import logging
import sys

def main(): 
    # Note: rs -> replisome location, determined by DnaB signal
    csv_path = 'C:\\Users\\noodl\\OneDrive\\Desktop\\MicroscopyTest\\Pr212dataSet\\AnalysisRebindCBCstart100noslow' # csv from trackmate
    rs_path = 'C:\\Users\\noodl\\OneDrive\\Desktop\\MicroscopyTest\\Pr212dataSet\\particles_result' # *.tif.RESULT
    mask_path = 'C:\\Users\\noodl\\OneDrive\\Desktop\\MicroscopyTest\\Pr212dataSet\\seg' # *.png

    # Additional Parameters
    rebind_only_particles = False # Skip bound outside particles for rebind event tabulation
    min_bound = 2 # min # frame for bound track
    min_lifetrack = 50 # min # frame for lifetime track
    max_frame_gap = 4 # frame, max frame gap allowed
    max_distance_gap = 2 # pix, max distance allowed for lifetime/bound difference
    entry_tolerance = 2 # frame, first n frames with double distance gap allowed
    overlap_tolerance = 10 # frame, max overlap for bound tracks allowed, removes the cell if exceeds
    box_size = 4 # pix, edge length for replisome box

    # Logging in both file and console
    log_file = csv_path + '\\_ColBD_LOG.txt'
    log_targets = [logging.FileHandler(log_file)]
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=log_targets)
    logging.StreamHandler.terminator = ''
    open(log_file, 'w').close()
    os.system('cls')

    # Debug Note: single-letter front -> file paths, double-letter front -> data
    # Debug Note: csv_sorted: [sB, sL, tB, tL] per entry
    csv_sorted = csv_name_sort_loose(csv_path)
    spotKeys = natsorted(list(csv_sorted.keys()))
    rS = get_rs_with_ext(rs_path, 'tif_Results.csv')
    masks = natsorted(get_file_names_with_ext(mask_path, 'png'))

    # masks = csv_mask_match(csv_sorted, spotKeys, masks)

    final_result = []
    final_rs = []
    final_rebind = []
    final_rebind_unsuccessful_count = 0
    final_bound_count = 0
    final_bound_failed_count = 0
    bound_track_total = 0

    # track classification`
    for i in range(len(masks)):
        print_log('Processing:', masks[i])
        mask = np.swapaxes(imgio.imread(masks[i]), 0, 1)
        n_cell = np.max(mask)
        rsS = rs_recognition(rS[i], mask, n_cell)

        sB = natsorted(csv_sorted[spotKeys[i]][0])
        sL = natsorted(csv_sorted[spotKeys[i]][1])
        sB = index_format(sB, n_cell, masks[i])
        sL = index_format(sL, n_cell, masks[i])
        # tB[i] = index_format(tB[i], n_cell)
        # tL[i] = index_format(tL[i], n_cell)
        
        for k in range(n_cell):
            print_log('\t-> Cell', k+1, ': ', end='')
            spotsBound = parse_csv(mask, sB[k], k + 1, is_spots=True)
            spotsLife = parse_csv(mask, sL[k], k + 1, is_spots=True)
            # tracksBound = parse_csv(mask, tB[i][k], k + 1, is_spots=False) # no need
            # tracksLife = parse_csv(mask, tL[i][k], k + 1, is_spots=False) # no need

            if(spotsLife == None):
                print_log('No Spots in Lifetime')
                continue # No spots
            trL = spots_to_tracks(spotsLife)

            if(spotsBound == None): # Label as Diffusive for all
                for entry in trL:
                    label_spots(trL[entry], (0,), -1)
                print_log('Only Diffusive Spots')
                final_bound_failed_count += 1
            else:
                n_fit = 0
                trB = spots_to_tracks(spotsBound)
                bound_track_total += len(list(trB.keys()))
                trL_aligned = track_align(trL, max_frame_gap, max_distance_gap)

                if(overlap_check(trB, overlap_tolerance)):
                    print_log('Overlap Exceeds')
                    continue

                fits_pending = []
                for entryB in trB:
                    trackBound = trB[entryB]
                    if len(trackBound) < min_bound:
                        continue
                    frB = [a[0] for a in trackBound]
                    frB_min, frB_max = frB[0], frB[-1]
                    fits = {}
                    for c in range(len(trL_aligned)):
                        trackLife = trL_aligned[c]
                        if((frB_max > trackLife[0][0] or frB_min < trackLife[-1][0]) and len(trackLife) > min_lifetrack):
                            fit = fit_bound_to_track(trackBound, trackLife, max_frame_gap, max_distance_gap, entry_tolerance)
                            if(fit == None): continue
                            if(rsS[k] == None):
                                rs_index = 0
                            else: 
                                rs_index = fit_track_to_rs(trackBound, rsS[k], box_size)
                            for entryF in fit:
                                if(len(entryF) > 4):
                                    entryF[4] = rs_index
                            fits[c] = fit.copy()
                            n_fit += 1
                    fits_pending.append(fits)
                trL = fits_on_track(fits_pending, trL_aligned, max_distance_gap)
                print_log('Track Fit to Lifetime (' + str(n_fit) + ')')

            # final_result + rebinding_tabulate
            t = 1
            for entryL in trL:
                entry = [i+1, k+1, t]
                trackLife = trL[entryL]
                rebinds, unsuccessful = rebind_record(trackLife, rebind_only_particles)
                final_rebind_unsuccessful_count += unsuccessful
                bound_count = bound_record(trackLife)
                if(bound_count > 0): final_bound_count += bound_count
                else: final_bound_failed_count += 1
                label_spots(trackLife, (0,), -1)
                for spot in trackLife:
                    formatted = entry.copy()
                    formatted += spot
                    final_result.append(formatted)
                for event in rebinds:
                    formatted = entry.copy()
                    formatted += event
                    final_rebind.append(formatted)
                t += 1

        # final_rs
        for r in range(len(rsS)):
            if(rsS[r] == None): continue
            entry = [i+1, r+1]
            s = 1
            for spot in rsS[r]:
                final_rs.append(entry + [s] + list(spot[2:]))
                s += 1

    print_log('# Track Bound: ', bound_track_total)

    # check
    for entry in final_result:
        if(not len(entry) == 9):
            print_log(entry)

    # Analysis
    print_log ('Analyzing: Dwell')
    final_dwell = dwell_isolation(final_result)
    final_dwell_not_bound = dwell_isolation(final_result, consider=-1)

    final_dwell_not_bound_long = []
    for entry in final_dwell_not_bound:
        if(entry[3] > 10):
            final_dwell_not_bound_long.append(entry)

    final_dwell_not_bound_short = []
    for entry in final_dwell_not_bound:
        if(entry[3] <= 10):
            final_dwell_not_bound_short.append(entry)
    
    print_log('\tBound to Fixed Particle:')
    print_log('\t\t-> # Track Bound =', len(final_dwell))
    print_log('\t\t-> Average Dwell Time:', np.mean([entry[3] for entry in final_dwell]))
    print_log('\t\t\tstd:', np.std([entry[3] for entry in final_dwell]))
    print_log('')
    print_log('\tAll Bound Tracks')
    print_log('\t\t-> # Track Bound =', len(final_dwell_not_bound))
    print_log('\t\t-> Average Dwell Time:', np.mean([entry[3] for entry in final_dwell_not_bound]))
    print_log('\t\t\tstd:', np.std([entry[3] for entry in final_dwell_not_bound]))
    print_log('')
    print_log('\tOnly Long Tracks > 10 frames')
    print_log('\t\t-> # Track Bound =', len(final_dwell_not_bound_long))
    print_log('\t\t-> Average Dwell Time:', np.mean([entry[3] for entry in final_dwell_not_bound_long]))
    print_log('\t\t\tstd:', np.std([entry[3] for entry in final_dwell_not_bound_long]))
    print_log('')
    print_log('\tOnly Short Tracks <= 10 frames')
    print_log('\t\t-> # Track Bound =', len(final_dwell_not_bound_short))
    print_log('\t\t-> Average Dwell Time:', np.mean([entry[3] for entry in final_dwell_not_bound_short]))
    print_log('\t\t\tstd:', np.std([entry[3] for entry in final_dwell_not_bound_short]))
    print_log('')
    print_log('------------------------------------------------------------------------------------------')
    print_log('')

    # Analysis Rebinding
    print_log('Analyzing: Rebinding Time')
    final_rebind_purge_single = []
    for entry in final_rebind:
        if(entry[-2] > 1):
            final_rebind_purge_single.append(entry)

    final_rebind_only_rs = []
    for entry in final_rebind:
        if(not(entry[-3] == 0 or entry[-4] == 0)):
            final_rebind_only_rs.append(entry)

    final_rebind_same = []
    final_rebind_diff = []
    for entry in final_rebind_only_rs:
        if(entry[-3] == entry[-4]):
            final_rebind_same.append(entry)
        else:
            final_rebind_diff.append(entry)
    
    print_log('\tRebinding to Everything | with single-frame events:')
    print_log('\t\t-> # Rebinding Event =', len(final_rebind))
    print_log('\t\t-> Average Rebind Time:', np.mean([entry[-2] for entry in final_rebind]))
    print_log('\t\t\tstd:', np.std([entry[-2] for entry in final_rebind]))
    print_log('')
    print_log('\tRebinding to Everything | without single-frame events:')
    print_log('\t\t-> # Rebinding Event =', len(final_rebind_purge_single))
    print_log('\t\t-> Average Rebind Time:', np.mean([entry[-2] for entry in final_rebind_purge_single]))
    print_log('\t\t\tstd:', np.std([entry[-2] for entry in final_rebind_purge_single]))
    print_log('')
    print_log('\tRebinding to Fixed Particles | with single-frame events:')
    print_log('\t\t-> # Rebinding Event =', len(final_rebind_only_rs))
    print_log('\t\t-> Average Rebind Time:', np.mean([entry[-2] for entry in final_rebind_only_rs]))
    print_log('\t\t\tstd:', np.std([entry[-2] for entry in final_rebind_only_rs]))
    print_log('')
    print_log('\tRebinding to Fixed Particles (SAME) | with single-frame events:')
    print_log('\t\t-> # Rebinding Event =', len(final_rebind_same))
    print_log('\t\t-> Average Rebind Time:', np.mean([entry[-2] for entry in final_rebind_same]))
    print_log('\t\t\tstd:', np.std([entry[-2] for entry in final_rebind_same]))
    print_log('')
    print_log('\tRebinding to Fixed Particles (DIFFERENT) | with single-frame events:')
    print_log('\t\t-> # Rebinding Event =', len(final_rebind_diff))
    print_log('\t\t-> Average Rebind Time:', np.mean([entry[-2] for entry in final_rebind_diff]))
    print_log('\t\t\tstd:', np.std([entry[-2] for entry in final_rebind_diff]))
    print_log('')
    print_log('------------------------------------------------------------------------------------------')
    print_log('')

    print_log('Analyzing: Rebinding Probability')
    final_rprob_all = float(len(final_rebind)) / float(len(final_rebind) + final_rebind_unsuccessful_count)
    final_rprob_purge_single = float(len(final_rebind_purge_single)) / float(len(final_rebind_purge_single) + final_rebind_unsuccessful_count)

    print_log('\tRebinding to Everything | with single-frame events:')
    print_log('\t\t-> # Rebinding Event =', len(final_rebind))
    print_log('\t\t-> # Unsuccessful:', final_rebind_unsuccessful_count)
    print_log('\t\t-> Probability:', final_rprob_all * 100, '%')
    print_log('')
    print_log('\tRebinding to Everything | without single-frame events:')
    print_log('\t\t-> # Rebinding Event =', len(final_rebind_purge_single))
    print_log('\t\t-> # Unsuccessful:', final_rebind_unsuccessful_count)
    print_log('\t\t-> Probability:', final_rprob_purge_single * 100, '%')
    print_log('')
    print_log('------------------------------------------------------------------------------------------')
    print_log('')

    print_log('Analyzing: Binding Probability')
    final_bprob_all = float(final_bound_count) / float(final_bound_count + final_bound_failed_count)

    print_log('\tRebinding to Everything | with single-frame events:')
    print_log('\t\t-> # Binding Event =', final_bound_count)
    print_log('\t\t-> # Unsuccessful:', final_bound_failed_count)
    print_log('\t\t-> Probability:', final_bprob_all * 100, '%')
    print_log('')
    print_log('------------------------------------------------------------------------------------------')
    print_log('')

    # output, fixed-particles and colocalized
    csv_write(csv_path + '\\_ColBD_fixed-particles.csv', final_rs)
    csv_write(csv_path + '\\_ColBD_spots.csv', final_result)
    csv_write(csv_path + '\\_ColBD_rebinding.csv', final_rebind)
    return

'''
================================================================================================================
ANALYSIS: BINDING
================================================================================================================
'''
def bound_record(track):
    bound_count = 0
    active = False
    record = track[0][4] if len(track[0]) > 4 else -1
    f = 0
    i = 1
    while(f < len(track)):
        if(len(track[f]) <= 4): active = True
        if(len(track[f]) > 4 and record == -1):
            if(active): bound_count += 1
            record = track[f][4]
        elif(len(track[f]) <= 4 and record > -1):
            record = -1
        elif(len(track[f]) > 4):
            record = track[f][4]
        f += 1

    return bound_count

'''
================================================================================================================
ANALYSIS: REBINDING
================================================================================================================
'''
# Analyze after each fit, tabulate all information about rebinding
# format:
#   [number, rs_prev, rs_next, rebinding time, avg speed (pix/fr)]
def rebind_record(track, skip0:bool):
    rebinds = []
    event = []
    active = False
    record = track[0][4] if len(track[0]) > 4 else -1
    f = 0
    i = 1
    
    # Treat 0 as diffusing
    if(skip0):
        track = track.copy()
        for i in range(len(track)):
            entry = track[i]
            if(len(entry) > 4):
                if(entry[4] == 0):
                    track[i] = entry[:4]
    
    while(f < len(track)):
            if(len(track[f]) > 4): active = True
            if(len(event) > 0 and len(track[f]) > 4):
                rebinds.append([i] + rebind_tabulate(event.copy(), record, track[f][4]))
                event = []
                record = track[f][4]
                i += 1
            elif(len(track[f]) <= 4 and active):
                event.append(track[f])
            elif(len(track[f]) > 4 and active):
                record = track[f][4]
            f += 1

    # unsuccessful event
    unsuccessful = 1 if (len(event) > 0) else 0
    return rebinds, unsuccessful

def rebind_tabulate(segment, prev, nxt):
    frames = [s[0] for s in segment]
    rebinding_time = max(frames) - min(frames) + 1
    distances = [0]
    if(len(segment) > 1):
        for i in range(1, len(segment)):
            distances.append(distance([segment[i-1][1], segment[i-1][2]], [segment[i][1], segment[i][2]]))
    return [prev, nxt, rebinding_time, sum(distances)/rebinding_time if (len(segment) > 1) else 1]

'''
================================================================================================================
ANALYSIS: DWELL
================================================================================================================
'''

# consider =0 not include 0, -1 include 0
def dwell_isolation(final_result, consider=0):
    final_dwell = []
    state = -1
    track = []
    for entry in final_result:
        if(entry[7] == state):
            if(len(track) > 0):
                if(track[0][2] != entry[2] or track[0][1] != entry[1]): # bound end
                    final_dwell.append(dwell(track))
                    track = []
                    state = entry[7]
                    continue
            if(state > consider):
                track.append(entry)
            continue
        if(state <= consider and entry[7] <= consider): continue # not consider localized to replisome
        if(state == -1 and entry[7] > consider):
            state = entry[7]
            track.append(entry)
        elif(state > consider): # bound entry
            if(len(track) > 0):
                final_dwell.append(dwell(track))
                state = entry[7]
                track = []
    if(len(track) > 0): final_dwell.append(dwell(track))
    return final_dwell

def dwell(track):
    frames = [entry[3] for entry in track]
    return [track[0][0], track[0][1], track[0][2], max(frames) - min(frames)]

'''
================================================================================================================
TRACK CLASSIFICATION
================================================================================================================
'''
# Checking overlaps in bound tracks
def overlap_check(trB:dict, overlap_tolerance):
    tracks = list(trB.values())
    frames = [[a[0] for a in tr] for tr in tracks]
    data = [(min(fr), max(fr)) for fr in frames]
    if(len(data) <= 1): return False
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if( data[i][1] < data[j][0] or data[j][1] < data[i][0]): continue
            if(min(data[i][1] - data[j][0], data[j][1] - data[i][0]) > overlap_tolerance):
                return True
    return False

# fits multiple tracks on lifetime tracks
# Under distance threshold, longer the better.
def fits_on_track(pending, tracklife, distance_gap):
    res = {}
    # eliminates with distance
    for tracks in pending:
        if(len(tracks.keys()) <= 1): continue
        deletion = []
        for key in tracks:
            # Max distance > gap
            if(fits_evaluate(tracks[key])[4] > distance_gap):
                deletion.append(key)
        for key in deletion:
            del tracks[key]
        

    # fits with resolved frame conflict
    for c in range(len(tracklife)):
        fits = []
        fits_parameters = []
        for tracks in pending:
            if(c in tracks.keys()):
                fits.append(tracks[c])
                fits_parameters.append(fits_evaluate(tracks[c]))
        
        trL = tracklife[c]
        frames = [l[0] for l in trL]
        frame_min, frame_max = min(frames), max(frames)
        frames = [-1]*(frame_max - frame_min + 1)
        selected = {}
        
        for i in range(len(fits)):
            fit = fits[i]
            do_insert = True
            fit_parameter = fits_parameters[i]
            if(fit_parameter[0] < frame_min or
               fit_parameter[1] > frame_max):
                continue
            for f in range(fit_parameter[0] - frame_min, fit_parameter[1] + 1 - frame_min):
                if(not frames[f] == -1):
                    compare = fits_parameters[frames[f]]
                    if(fits_compare(fit_parameter, compare)):
                        do_insert = False
                        break
                    else:
                        del selected[frames[f]]
                        fits_remove(frames, frames[f])
            if(do_insert):
                fits_choose(frames, fit_parameter[0] - frame_min, fit_parameter[1] - frame_min, i)
                selected[i] = fit
        
        # insert tracks
        for key in selected:
            trL = track_insert(trL, selected[key])
        res[c]= trL
    return res

def fits_choose(frames, fmin, fmax, index):
    for i in range(fmin, fmax + 1):
        frames[i] = index

def fits_remove(frames, index):
    for i in range(len(frames)):
        if(frames[i] == index):
            frames[i] = -1

# Adjustable comparison
def fits_compare(parameter1, parameter2):
    # compare length
    if(parameter1[2] < parameter2[2]): return True
    else: return False

def fits_evaluate(track):
    frames = [t[0] for t in track]
    fmin, fmax = min(frames), max(frames)
    flength = fmax - fmin
    distances = [t[5] for t in track]
    dmax, dmean = max(distances), np.mean(distances)
    return fmin, fmax, flength, dmean, dmax

def track_anneal(trackf, trackb):
    end = trackf[-1][0]
    index = 0
    while(trackb[index][0] <= end): index += 1
    if(index >= len(trackb)):
        return trackf
    return trackf + trackb[index:]

def track_align(tracks_raw, frame_gap, distance_gap):
    tracks = []
    for entryT in tracks_raw:
        track = tracks_raw[entryT]
        frames = [a[0] for a in track]
        tracks.append((track.copy(), min(frames), max(frames)))
    tracks.sort(key=lambda x: x[1])
    
    res = []
    queue = [(0, tracks[0])]
    while(len(queue) > 0):
        root = queue.pop(0)
        start = root[0] + 1
        track = root[1]
        if(start >= len(tracks)):
            res.append(track[0])
            break
        for i in range(start, len(tracks)):
            compare = tracks[i]
            if(abs(compare[1] - track[2]) > frame_gap or 
               distance([track[0][-1][1], track[0][-1][2]], [compare[0][0][1], compare[0][0][2]]) > distance_gap):
                exist = False
                for item in queue:
                    if(item[0] == i):
                        exist = True
                        break
                if(not exist): queue.append((i, compare))
            else:
                track = (track_anneal(track[0], compare[0]), track[1], track[2])
        res.append(track[0])
    return res

def csv_write(path, data):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for line in data:
            writer.writerow(line)
        file.close()

def track_insert(track, insert):
    frames = [entry[0] for entry in insert]
    res = []
    inserted = False
    for entry in track:
        if(entry[0] < min(frames) or entry[0] >= max(frames)): res.append(entry)
        else:
            if(not inserted):
                res += insert
                inserted = True
    return res

# naive w/ respect to time
def fit_bound_to_track(bound, life, frame_gap, distance_gap, entry_tolerance):
    # check if contains matching frames
    if(
        life[0][0] - bound[-1][0] > 0 or
        bound[0][0] - life[-1][0] > 0
    ): return None

    # find starting point for fitting
    l = 0 # lifetime array index
    b = 0 # bound array index
    while(life[l][0] < bound[b][0]): l += 1
    if(l > 0 and bound[b][0] - life[l-1][0] <= frame_gap): l = l - 1
    frame = min(life[l][0], bound[b][0])
    fit = []
    gap = 0
    record = (bound[b][1], bound[b][2])
    while True:
        if(b >= len(bound) or l >= len(life)): break
        if(life[l][0] != frame and bound[b][0] != frame):
            if(gap > frame_gap): break
            gap += 1
            continue
        elif(life[l][0] != frame):
            d = distance((bound[b][1], bound[b][2]), (record[0], record[1]))
            if(d > distance_gap):
                gap += 1
                if(gap > frame_gap): break
            else: gap = 0
            record = (bound[b][1], bound[b][2])
            entry = bound[b].copy()
            if(len(entry) > 5): return None
            entry.append(-10) # placeholder
            entry.append(d)
            fit.append(entry)
            b += 1
        elif(bound[b][0] != frame):
            d = distance((life[l][1], life[l][2]), (record[0], record[1]))
            if(d > distance_gap):
                gap += 1
                if(gap > frame_gap): break
            else: gap = 0
            record = (life[l][1], life[l][2])
            entry = life[l].copy()
            if(len(entry) > 5): return None
            entry.append(-10) # placeholder
            entry.append(d)
            fit.append(entry)
            l += 1
        else:
            d1 = distance((life[l][1], life[l][2]), (record[0], record[1]))
            d2 = distance((life[l][1], life[l][2]), (bound[b][1], bound[b][2]))
            if(((d1 > distance_gap or d2 > distance_gap) and b > entry_tolerance) or 
                    (d1 > distance_gap * 2 or d2 > distance_gap * 2)):
                break
            else: gap = 0
            record = (life[l][1], life[l][2])
            entry = life[l].copy()
            if(len(entry) > 5): return None
            entry.append(-10) # placeholder
            entry.append(d2)
            fit.append(entry)
            l += 1
            b += 1
        frame += 1

    if(len(fit) < 1): return None
    return fit

def distance(p1, p2):
    return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1] - p2[1], 2))

def fit_track_to_rs(spots, rs, box_size):
    if(rs == None): return 0
    x, y = track_bound_avg(spots)
    # 7, 9; 8, 10; for rs location min,max
    for i in range(len(rs)):
        r = rs[i]
        rx = r[2]
        ry = r[3]
        rxmin, rxmax = rx - box_size/2, rx + box_size/2
        rymin, rymax = ry - box_size/2, ry + box_size/2
        if((x >= rxmin and x <= rxmax) and
            (y >= rymin and y <= rymax)): 
            return i+1
    return 0
    
def track_bound_avg(spots):
    mean = np.mean(np.array(spots), axis=0)
    return (mean[1], mean[2])

# label spots in track, list position (not frame number)
def label_spots(track, indices:tuple, label:int):
    if(len(indices) == 1):
        for i in range(indices[0], len(track)):
            if(len(track[i]) <= 4): track[i] += [label, -1]
    elif(len(indices) == 2):
        for i in range(indices[0], indices[1]):
            if(len(track[i]) <= 4): track[i] += [label, -1]

# convert spots to dictionary of tracks
def spots_to_tracks(spots):
    if(len(spots) < 2): return None
    tracks = {}

    k = 0
    data = spots[k]
    track_number = int(data[0])
    track = []
    while(k < len(spots)):
        data = spots[k]
        n = int(data[0])
        if (track_number != n):
            track = sort_by_frame(track)
            tracks[track_number] = track.copy()
            track = []
            track_number = n
        track.append([int(data[1]), data[2], data[3], int(data[4])])
        k += 1
    track = sort_by_frame(track)
    tracks[track_number] = track.copy()
    return tracks

'''
================================================================================================================
INPUT FORMATTING
================================================================================================================
'''
# In case some videos have no tracks
def csv_mask_match(csv, videos, masks):
    mnames = {}
    vnames = {}
    for m in masks:
        name = m.split('\\')[-1].split('_')
        index = -1
        for n in range(len(name)):
            try:
                if(int(name[n]) >= 0):
                    index = n + 1
                    break
            except:
                continue
        number = int(name[index])
        name = '_'.join(name[:index])
        if(name in mnames):
            mnames[name].append(number)
        else:
            mnames[name] = [number]
    for v in videos:
        name = v.split('\\')[-1].split('_')
        index = -1
        for n in range(len(name)):
            try:
                if(int(name[n]) >= 0):
                    index = n + 1
                    break
            except:
                continue
        number = int(name[index])
        name = '_'.join(name[:index])
        if(name in vnames):
            vnames[name].append(number)
        else:
            vnames[name] = [number]
    return 1

def sort_by_frame(track):
    return sorted(track, key=lambda x: x[0])

# Tabulate replisome locations with cell #
def rs_recognition(file, mask, n_cell):
    if file == None: return None
    data = np.loadtxt(file, delimiter=',', dtype=float, skiprows=1)
    if data.ndim == 1:
        data = np.array([data])

    res = [[] for i in range(n_cell)]
    
    # ,Abs_frame,X_(px),Y_(px),Channel,Slice,Frame,xMin,yMin,xMax,yMax,NArea,IntegratedInt
    for particle in data:
        rx = int(round(particle[2]))
        ry = int(round(particle[3]))
        try:
            cell = mask[rx, ry]
            if cell > 0: res[cell - 1].append(particle)
        except ValueError:
            continue
    return res

# separate files by cell #
def index_format(files, max, mask_name):
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

def parse_csv(mask, csv, index, is_spots):
    if csv == None:
        return None
    data = np.loadtxt(csv, delimiter=',', dtype=float)
    
    if data.ndim == 1:
        data = np.array([data])

    res = []
    
    for i in range(data.shape[0]):
        if is_spots:
            x = int(round(data[i][2]))
            y = int(round(data[i][3]))
        else:
            x = int(round(data[i][16]))
            y = int(round(data[i][17]))
        try:
            cell = mask[x, y]
            if(cell == index): res.append(data[i])
        except ValueError:
            continue
    return res

def get_file_names_with_ext(path:str, ext:str):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fname = file.split('.')
            if(fname[-1] == ext):
                flist.append(root + '\\' +  file)
    return flist

def get_rs_with_ext(path:str, ext:str):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fname = file.split('.')
            if(len(fname) < 2):
                continue
            if('.'.join([fname[-2], fname[-1]]) == ext):
                flist.append(root + '\\' +  file)
    return flist

def csv_name_sort_loose(path:str) -> dict:
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
        if(not video in csv_sorted):
            csv_sorted[video] = [[],[],[],[]]
        if 'spotsBound' in fname[ind+2]:
            csv_sorted[video][0].append(file)
        elif 'spotsAll' in fname[ind+2]:
            csv_sorted[video][1].append(file)
        elif 'tracksBound' in fname[ind+2]:
            csv_sorted[video][2].append(file)
        elif 'tracksAll' in fname[ind+2]:
            csv_sorted[video][3].append(file)

    return csv_sorted

# tired of typing for loops
def csv_name_sort_helper(fr):
    video = []

    for file in fr:
        fname = file.split('\\')[-1].split('_')
        video_name = "_".join(fname[:-3])
        if not (video_name in video):
            video.append(video_name)
    
    temp = []
    for i in range(len(video)):
        list = []
        temp.append(list)
    
    for file in fr:
        fname = file.split('\\')[-1].split('_')
        index1 = video.index("_".join(fname[:-3]))
        index2 = int(fname[-2]) - 1
        temp[index1].append(file)
    return temp

# Bumped by 1 to match .png format (0 is background)
def generate_indices(n_cell:int):
    indices = []
    for i in range(n_cell):
        indices.append(i+1)
    return indices

'''
================================================================================================================
START
================================================================================================================
'''

# Modified print
def print_log(*args, end='\n'):
    print(' '.join([str(a) for a in args]), end=end)
    logging.info(' '.join([str(a) for a in args] + [end]))

# Start Script
if __name__== '__main__':
	start_time = time.time()
	main()
	print_log("--- %s seconds ---" % (time.time() - start_time))