import numpy as np
from scipy import io as scio
import os
from natsort import natsorted
import time

def main(): 
    csv_path = 'C:\\Users\\JpRas\\OneDrive\\Escritorio\\RODREBIN\\wt\\timelapse\\AnalysisRebindCBC_start0_Quality5\\_ColBD_LIFE\\SMAUG_REBINDING_SPOTS'
    rA, rD, rS, rT, sA, sD, sS, sT = csv_name_sort(csv_path)
    common_name= 'DnaQssb113'
    timeinterval= '20m100i'
    array = parse_csv(rA, 1)
    scio.savemat(csv_path + '\\_SMAUG_'+timeinterval+'_'+common_name+'_relaxed_rebinds_spotsAll.mat', {'trfile':array})
    array = parse_csv(rD, 1)
    scio.savemat(csv_path + '\\_SMAUG_'+timeinterval+'_'+common_name+'_relaxed_rebinds_spotsDiff.mat', {'trfile':array})
    array = parse_csv(rS, 1)
    scio.savemat(csv_path + '\\_SMAUG_'+timeinterval+'_'+common_name+'_relaxed_rebinds_spotsSame.mat', {'trfile':array})
    array = parse_csv(rT, 1)
    scio.savemat(csv_path + '\\_SMAUG_'+timeinterval+'_'+common_name+'_relaxed_rebinds_spotsTrack.mat', {'trfile':array})

    array = parse_csv(sA, 1)
    scio.savemat(csv_path + '\\_SMAUG_'+timeinterval+'_'+common_name+'_strict_rebinds_spotsAll.mat', {'trfile':array})
    array = parse_csv(sD, 1)
    scio.savemat(csv_path + '\\_SMAUG_'+timeinterval+'_'+common_name+'_strict_rebinds_spotsDiff.mat', {'trfile':array})
    array = parse_csv(sS, 1)
    scio.savemat(csv_path + '\\_SMAUG_'+timeinterval+'_'+common_name+'_strict_rebinds_spotsSame.mat', {'trfile':array})
    array = parse_csv(sT, 1)
    scio.savemat(csv_path + '\\_SMAUG_'+timeinterval+'_'+common_name+'_relaxed_rebinds_spotsTrack.mat', {'trfile':array})


def parse_csv(sL, placeholder):
    index = 1
    res = []
    
    for flist in sL:
        for fname in flist:
            print('Processing -> ' + fname)
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
                if(tn_new == tn):
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

def get_file_names_with_ext(path:str, ext:str):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fname = file.split('.')
            if(fname[-1] == ext):
                flist.append(root + '\\' +  file)
    return flist

def csv_name_sort(path:str):
    flist = get_file_names_with_ext(path, 'csv')
    rA = []
    rD = []
    rS = []
    rT = []
    sA = []
    sD = []
    sS = []
    sT = []
    for file in flist:
        fname = file.split('\\')[-1]
        print(fname)
        #if len(fname) < 4:
        #    continue
        #if not fname[-3] == 'Cell':
        #    continue
        if fname == 'relaxed_rebinds_spotsAll.csv':
            rA.append(file)
        elif fname == 'relaxed_rebinds_spotsDiff.csv':
            rD.append(file)
        elif fname == 'relaxed_rebinds_spotsSame.csv':
            rS.append(file)
        elif fname == 'relaxed_rebinds_spotsTrack.csv':
            rT.append(file)
        elif fname == 'strict_rebinds_spotsAll.csv':
            sA.append(file)
        elif fname == 'strict_rebinds_spotsDiff.csv':
            sD.append(file)
        elif fname == 'strict_rebinds_spotsSame.csv':
            sS.append(file)
        elif fname == 'strict_rebinds_spotsTrack.csv':
            sT.append(file)



    return [csv_name_sort_helper(natsorted(rA)), 
            csv_name_sort_helper(natsorted(rD)),
            csv_name_sort_helper(natsorted(rS)),
            csv_name_sort_helper(natsorted(rT)),
            csv_name_sort_helper(natsorted(sA)),
            csv_name_sort_helper(natsorted(sD)),
            csv_name_sort_helper(natsorted(sS)),
            csv_name_sort_helper(natsorted(sT)),
            ]


# tired of typing for loops
def csv_name_sort_helper(fr):
    video = []

    for file in fr:
        fname = file.split('\\')[-1].split('.')
        video_name = "_".join(fname[:-2])
        if not (video_name in video):
            video.append(video_name)
    
    temp = []
    for i in range(len(video)):
        list = []
        temp.append(list)
    
    for file in fr:
        fname = file.split('\\')[-2].split('.')
        index1 = video.index("_".join(fname[:-2]))
        #index2 = int(fname[-2].split('.')[0]) - 1
        #index2 = int(fname[-2]) - 1
        temp[index1].append(file)
    return temp

if True:
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))