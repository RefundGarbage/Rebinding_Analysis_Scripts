import numpy as np
from scipy import io as scio
from skimage import io as imgio
import os
from natsort import natsorted
import time
import csv

def main(): 
    # Note: rs -> replisome location, determined by DnaB signal
    csv_path = 'F:\\DiffusionAnalysis\\Pr212dataSet\\AnalysisRebindCBCstart100noslow' # 
    rs_path = 'F:\\DiffusionAnalysis\\Pr212dataSet\\particles_result' # *.tif.RESULT
    mask_path = 'F:\\DiffusionAnalysis\\Pr212dataSet\\seg' # *.png
    max_frame_gap = 4
    max_distance_gap = 4
    entry_tolerance = 2
    box_size = 4

    # Debug Note: single-letter front -> file paths, double-letter front -> data
    sB, sL, tB, tL = csv_name_sort_loose(csv_path)
    rS = get_rs_with_ext(rs_path, 'tif_Results.csv')
    masks = natsorted(get_file_names_with_ext(mask_path, 'png'))

    final_result = []
    final_rs = []

    # track classification`
    for i in range(len(masks)):
        print('Processing:', masks[i])
        mask = np.swapaxes(imgio.imread(masks[i]), 0, 1)
        n_cell = np.max(mask)
        rsS = rs_recognition(rS[i], mask, n_cell)

        sB[i] = index_format(sB[i], n_cell)
        sL[i] = index_format(sL[i], n_cell)
        # tB[i] = index_format(tB[i], n_cell)
        # tL[i] = index_format(tL[i], n_cell)
        
        for k in range(n_cell):
            print('\t-> Cell', k+1, ': ', end='')
            spotsBound = parse_csv(mask, sB[i][k], k + 1, is_spots=True)
            spotsLife = parse_csv(mask, sL[i][k], k + 1, is_spots=True)
            # tracksBound = parse_csv(mask, tB[i][k], k + 1, is_spots=False) # no need
            # tracksLife = parse_csv(mask, tL[i][k], k + 1, is_spots=False) # no need

            if(spotsLife == None):
                print('No Spots in Lifetime')
                continue # No spots
            trL = spots_to_tracks(spotsLife)

            if(spotsBound == None): # Label as Diffusive for all
                for entry in trL:
                    label_spots(trL[entry], (0,), -1)
                print('Only Diffusive Spots')
            else:
                n_fit = 0
                trB = spots_to_tracks(spotsBound)
                trL_aligned = track_align(trL, max_frame_gap, max_distance_gap)
                for entryB in trB:
                    trackBound = trB[entryB]
                    frB = [a[0] for a in trackBound]
                    frB_min, frB_max = frB[0], frB[-1]
                    for c in range(len(trL_aligned)):
                        trackLife = trL_aligned[c]
                        if(frB_max > trackLife[0][0] or frB_min < trackLife[-1][0]):
                            fit = fit_bound_to_track(trackBound, trackLife, max_frame_gap, max_distance_gap, entry_tolerance)
                            if(fit == None): continue
                            if(rsS[k] == None):
                                rs_index = 0
                            else: 
                               rs_index = fit_track_to_rs(trackBound, rsS[k], box_size)
                            for entryF in fit:
                                if(len(entryF) > 4):
                                    entryF[4] = rs_index
                            trL_aligned[c] = track_insert(trackLife, fit)
                            n_fit += 1
                            break
                trL = {}
                for c in range(len(trL_aligned)):
                    trL[c] = trL_aligned[c]
                print('Track Fit to Lifetime (' + str(n_fit) + ')')
            
            # final_result
            t = 1
            for entryL in trL:
                entry = [i+1, k+1, t]
                trackLife = trL[entryL]
                label_spots(trackLife, (0,), -1)
                for spot in trackLife:
                    formatted = entry.copy()
                    formatted += spot
                    final_result.append(formatted)
                t += 1

        # final_rs
        for r in range(len(rsS)):
            if(rsS[r] == None): continue
            entry = [i+1, r+1]
            s = 1
            for spot in rsS[r]:
                final_rs.append(entry + [s] + list(spot[2:]))
                s += 1

    # Analysis
    print ('Analyzing: Dwell')
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
            if(state > 0):
                track.append(entry)
            continue
        if(state <= 0 and entry[7] <= 0): continue # not consider localized to replisome
        if(state == -1 and entry[7] > 0):
            state = entry[7]
            track.append(entry)
        elif(state > 0): # bound end
            final_dwell.append(dwell(track))
            state = entry[7]
            track = []
    if(len(track) > 0): final_dwell.append(dwell(track))
    
    print('\t-> # Track Bound =', len(final_dwell))
    print('\t-> Average Dwell Time:', np.mean([entry[3] for entry in final_dwell]))

    # output, fixed-particles and colocalized
    csv_write(csv_path + '\\_ColBD_fixed-particles.csv', final_rs)
    csv_write(csv_path + '\\_ColBD_spots.csv', final_result)
    return

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

def track_anneal(trackf, trackb):
    end = trackf[-1][0]
    index = 0
    while(trackb[index][0] <= end): index += 1
    if(index >= len(trackb)):
        return trackf
    return trackf + trackb[index:]

def dwell(track):
    frames = [entry[3] for entry in track]
    return [track[0][0], track[0][1], track[0][2], max(frames) - min(frames)]

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
            entry.append(-10) # placeholder
            entry.append(d)
            fit.append(entry)
            l += 1
        else:
            d1 = distance((life[l][1], life[l][2]), (record[0], record[1]))
            d2 = distance((life[l][1], life[l][2]), (bound[b][1], bound[b][2]))
            if((d1 > distance_gap or d2 > distance_gap) and b > entry_tolerance):
                break
            else: gap = 0
            record = (life[l][1], life[l][2])
            entry = life[l].copy()
            entry.append(-10) # placeholder
            entry.append(d1)
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
def index_format(files, max):
    res = [None]*max
    for file in files:
        index = index_find(file)
        if(not index == -1):
            res[index - 1] = file
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

def csv_name_sort_loose(path:str):
    flist = get_file_names_with_ext(path, 'csv')
    sB = []
    sL = []
    tB = []
    tL = []
    for file in flist:
        fname = file.split('\\')[-1].split('_')
        if len(fname) < 4:
            continue
        if not fname[-3] == 'Cell':
            continue
        if 'spotsBound' in fname[-1]:
            sB.append(file)
        elif 'spotsAll' in fname[-1]:
            sL.append(file)
        elif 'tracksBound' in fname[-1]:
            tB.append(file)
        elif 'tracksAll' in fname[-1]:
            tL.append(file)

    return [csv_name_sort_helper(natsorted(sB)), 
            csv_name_sort_helper(natsorted(sL)),
            csv_name_sort_helper(natsorted(tB)), 
            csv_name_sort_helper(natsorted(tL))]

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

if True:
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))