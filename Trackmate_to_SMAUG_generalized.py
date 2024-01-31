import numpy as np
from scipy import io as scio
from skimage import io as imgio
import os
from natsort import natsorted
import time

def main():
    csv_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\Tracking\\_ColBD_LIFE\\SMAUG_REBINDING_SPOTS'

    # Enter all suffixes used, files will be generated for each suffix
    # Note: At least one spots file per video must exist for all suffixes, otherwise errors will occur
    #       If only a few video have spots with the suffix, create separate folders for masks and spots
    suffixes = ['spotsAll', 'spotsTrack', 'spotsDiff', 'spotsSame']
    common_info = 'ColBD_LIFE'

    track_spots = {}
    for suffix in suffixes:
        track_spots[suffix] = csv_name_sort(csv_path, suffix)

    print(track_spots)

    for suffix in track_spots:
        tracks = track_spots[suffix]
        print('-'*len(suffix))
        print(suffix)
        print('-'*len(suffix))
        scio.savemat(csv_path + '\\_SMAUG_' + common_info + '_' + suffix + '.mat',
                    {'trfile': parse_csv(tracks, 1)})

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

    return natsorted(s)

if True:
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
