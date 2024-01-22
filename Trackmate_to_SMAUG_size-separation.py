import numpy as np
from scipy import io as scio
from skimage import io as imgio
import os
from natsort import natsorted
import time

def main():
    csv_path = 'F:\\MicroscopyTest\\Viri_SMAUG_size\\Images\\AnalysisRebindCBC'
    mask_path = 'F:\\MicroscopyTest\\Viri_SMAUG_size\\cellpose'

    # Enter all suffixes used, files will be generated for each suffix
    # Note: At least one spots file per video must exist for all suffixes, otherwise errors will occur
    #       If only a few video have spots with the suffix, create separate folders for masks and spots
    suffixes = ['spotsAll']
    common_info = 'VJ73_100msAFAP'

    size_partitions = [30, 60, 90] # pixels, # bins = length + 1

    masks = natsorted(get_file_names_with_ext(mask_path, 'png'))
    track_spots = {}
    for suffix in suffixes:
        track_spots[suffix] = csv_name_sort(csv_path, suffix)

    print(track_spots)
    print(masks)

    for suffix in track_spots:
        tracks = track_spots[suffix]
        print('-'*len(suffix))
        print(suffix)
        print('-'*len(suffix))

        # specific for size
        arrays = process_tracks(tracks, masks, size_partitions)
        for i in range(len(arrays)):
            if len(arrays[i]) > 0:
                scio.savemat(csv_path + '\\_SMAUG_' + common_info + '_' + suffix + '_size_' + str(i+1) + '.mat',
                            {'trfile': arrays[i]})

    # s0, s1, s2, s3, s4 = csv_name_sort(csv_path)
    # array = parse_csv(s0, 1)
    # scio.savemat(csv_path + '\\_SMAUG_' + timeinterval + '_' + common_name + '_All0.mat', {'trfile': array})
    # array = parse_csv(s1, 1)
    # scio.savemat(csv_path + '\\_SMAUG_' + timeinterval + '_' + common_name + '_All5.mat', {'trfile': array})
    # array = parse_csv(s2, 1)
    # scio.savemat(csv_path + '\\_SMAUG_' + timeinterval + '_' + common_name + '_Bound.mat', {'trfile': array})

def process_tracks(tracks, masks, size_partitions):
    track_sizes = []
    for i in range(len(size_partitions) + 1): track_sizes.append([])
    result = []

    for i in range(len(masks)):
        print('Processing: ' + masks[i])

        mask = np.swapaxes(imgio.imread(masks[i]), 0, 1)
        n_cell = np.max(mask)
        track_cells = index_format(tracks[i], n_cell)
        sizes = get_cell_sizes(mask, n_cell)

        for k in range(len(track_cells)):
            if(track_cells[k] == None): continue
            size = sizes[k]
            partition = partition_determine(size, size_partitions)
            track_sizes[partition].append(track_cells[k])
    for i in range(len(track_sizes)):
        print('Partition: ' + str(i+1))
        if(len(track_sizes[i]) == 0):
            result.append([])
            continue
        result.append(parse_csv(track_sizes[i], 1))
    return result

# Determine the partition of the cell given size
def partition_determine(size, partitions):
    i = 0
    while partitions[i] < size:
        i += 1
        if(i >= len(partitions)): break
    return i

def parse_csv(sL, placeholder):
    index = 1
    res = []

    for fname in sL:
        print('\t-> ' + fname)
        data = np.loadtxt(fname, delimiter=',', dtype=float)
        if len(data) == 0:
            continue
        try:
            tn = int(data[0][0])
        except:
            continue
        temp = []
        for line in data:
            tn_new = int(line[0])
            if (tn_new == tn):
                temp.append([index, int(line[1]), placeholder, line[2], line[3]])
            else:
                index += 1
                tn = tn_new
                temp.sort(key=lambda x: x[1])
                temp.sort(key=lambda x: x[0])
                start = temp[0][1]
                for i in range(len(temp)):
                    temp[i][1] = temp[i][1] - start + 1
                res += temp.copy()
                temp = [[index, int(line[1]), placeholder, line[2], line[3]]]
        index += 1
        temp.sort(key=lambda x: x[1])
        temp.sort(key=lambda x: x[0])
        start = temp[0][1]
        for i in range(len(temp)):
            temp[i][1] = temp[i][1] - start + 1
        res += temp.copy()
    return np.array(res)


def get_file_names_with_ext(path: str, ext: str):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fname = file.split('.')
            if (fname[-1] == ext):
                flist.append(root + '\\' + file)
    return flist


def csv_name_sort(path: str, suffix:str):
    flist = get_file_names_with_ext(path, 'csv')
    s = []
    for file in flist:
        fname = file.split('\\')[-1].split('_')
        # if len(fname) < 4:
        #    continue
        # if not fname[-3] == 'Cell':
        #    continue
        if fname[-1] == suffix + '.csv':
            s.append(file)

    return csv_name_sort_helper(natsorted(s))


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
        index2 = int(fname[-2].split('.')[0]) - 1
        # index2 = int(fname[-2]) - 1
        temp[index1].append(file)
    return temp

def get_cell_sizes(mask, n_cell):
    sizes = [0]*n_cell
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[x][y] > 0:
                sizes[mask[x][y] - 1] += 1
    return sizes

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
def generate_indices(n_cell:int):
    indices = []
    for i in range(n_cell):
        indices.append(i+1)
    return indices

if True:
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
