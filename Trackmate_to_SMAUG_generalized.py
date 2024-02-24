import numpy as np
from scipy import io as scio
from skimage import io as imgio
import os
from natsort import natsorted
import time

def main():

    common_info = 'DnaQwt_20m100i'

    csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\AnalysisRebindCBC_start0_Quality5'

    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\AnalysisRebindCBC_start0_Quality5'  #csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA7\\timelapse\\AnalysisRebindCBC_start0_Quality5'
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA3\\Timelapse\\AnalysisRebindCBC_start0_Quality5'  # csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\SSB113\\timelapse'
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA5\\timelapse'

    #csv_path = 'F:\\_Microscopy\\Rawdates\\_RUNreb1108_4diameter\\timelapse\\GOOD_OLD_Quality2point5\\pr208\\pr208_AnalysisRebindCBC1114_6diam_Dog'
    #csv_path = 'F:\\_Microscopy\\Rawdates\\_RUNreb1108_4diameter\\timelapse\\GOOD_OLD_Quality2point5\\pr212\\AnalysisRebindCBC_11146dia_dog'
    #csv_path = 'F:\\_Microscopy\\Rawdates\\_RUNreb1108_4diameter\\timelapse\\GOOD_OLD_Quality2point5\\pr208n\\AnalysisRebindCBC_11146diam_DOG'  # csv from trackmate

    #csv_path = 'F:\\_Microscopy\\Rawdates\\20230913_ypetB_haloQ\\Images\\timelapse\\101023\\AnalysisRebindCBC_1010123_start0'  # csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\set3'
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA3\\Timelapse\\set2'  # csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\MutA7\\timelapse\\set2'
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\AnalysisRebindCBC_start0_Quality5'

    #csv_path = 'F:\\_Microscopy\\Rawdates\\202301109_TetR_haloQmutnWT\\Images\\COMB\\AnalysisRebindCBC_5quality\\PR208' # csv from trackmate
    #csv_path = 'F:\\_Microscopy\\Rawdates\\202301109_TetR_haloQmutnWT\\Images\\COMB\\AnalysisRebindCBC_5quality\\PR212' # csv from trackmate
    #csv_path = 'F:\\_Microscopy\\Rawdates\\202309026_pr208DmHQctd\\Images\\AnalysisRebindCBCstart0'  # csv from trackmate
    #csv_path = 'F:\\_Microscopy\\Rawdates\\_RUNreb1108_4diameter\\timelapse\\GOOD_OLD_Quality2point5\\pr212\\AnalysisRebindCBC_11146dia_dog'  # csv from trackmate
    #csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\SSB113\\timelapse'  # csv from trackmate

    csv_path = csv_path+ '\\_ColBD_LIFE_FInal\\SMAUG_REBINDING_SPOTS'

    # Enter all suffixes used, files will be generated for each suffix
    # Note: At least one spots file per video must exist for all suffixes, otherwise errors will occur
    #       If only a few video have spots with the suffix, create separate folders for masks and spots
    suffixes = ['relaxed_rebinds_spotsAll',
                'relaxed_rebinds_spotsTrack',
                'strict_rebinds_spotsAll',
                'strict_rebinds_spotsTrack',
                'strict_rebinds_spotsDiff',
                'strict_rebinds_spotsSame']


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
    suffix = suffix.split('_')
    suffix[-1] += '.csv'
    left_index = -1 * len(suffix)
    s = []
    for file in flist:
        fname = file.split('\\')[-1].split('_')
        # if len(fname) < 4:
        #    continue
        # if not fname[-3] == 'Cell':
        #    continue
        if fname[left_index:] == suffix:
            s.append(file)

    return natsorted(s)

if True:
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
