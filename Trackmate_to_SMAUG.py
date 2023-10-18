import numpy as np
from scipy import io as scio
import os
from natsort import natsorted
import time

def main(): 
    csv_path = 'F:\\OlympusAnalysis\\test\\Diffusion_Test'
    sB, sL, tB, tL = csv_name_sort(csv_path)
    array = parse_csv(sL, 1)
    scio.savemat(csv_path + '\\SMAUG_tracks.mat', {'SMAUG_tracks':array})

def parse_csv(sL, placeholder):
    index = 1
    res = []
    
    for flist in sL:
        for fname in flist:
            print('Processing -> ' + fname)
            data = np.loadtxt(fname, delimiter=',', dtype=float)
            if len(data) == 0:
                continue
            tn = int(data[0][0])
            temp = []
            for line in data:
                tn_new = int(line[0])
                if(tn_new == tn):
                    temp.append([index, int(line[1]), placeholder, line[2], line[3]])
                else:
                    index += 1
                    tn = tn_new
                    temp.sort(key=lambda x: x[1])
                    for i in range(len(temp)):
                        temp[i][1] = i + 1
                    res += temp
                    temp = [[index, int(line[1]), placeholder, line[2], line[3]]]
            index += 1
            temp.sort(key=lambda x: x[3])
            res += temp
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
        if fname[-1] == 'spotsBound.csv':
            sB.append(file)
        elif fname[-1] == 'spotsLifetime.csv':
            sL.append(file)
        elif fname[-1] == 'tracksBound.csv':
            tB.append(file)
        elif fname[-1] == 'tracksLifetime.csv':
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

if True:
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))