import numpy as np
import pandas as pd
from skimage import io as imgio
import time
import os
from natsort import natsorted
import logging
import shutil
import tomllib

def main():

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config_path = os.path.join(__location__, 'script-config.toml')
    with open(config_path, 'rb') as config_file:
        configs = tomllib.load(config_file)

    csv_path = configs['path']['csv_path']
    mask_path = configs['path']['mask_path']
    output_folder_name = configs['path']['output_folder_name']

    masks = natsorted(get_file_names_with_ext(mask_path, 'png'))
    table = []
    columns = ['Mask #', 'Mask Name', '# Cells', 'Cell', 'Area', 'Length']
    for i in range(len(masks)):
        mask = np.swapaxes(imgio.imread(masks[i]), 0, 1)
        n_cell = np.max(mask)
        sizes = np.unique(mask, return_counts=True)
        cells = tabulate_cells(mask, n_cell)
        for j in range(n_cell):
            cell = np.array(cells[j])
            minx, maxx, miny, maxy = min(cell[:, 0]), max(cell[:, 0]), min(cell[:, 1]), max(cell[:, 1])
            l = np.sqrt(np.power(maxx-minx+1, 2) + np.power(maxy-miny+1, 2))
            table.append([i + 1, masks[i].split('\\')[-1], n_cell, j + 1, sizes[1][j + 1], l])
    table = pd.DataFrame(table, columns=columns)
    table.to_csv(mask_path + '\\_cell-info.csv')
    table.to_csv(csv_path + "\\" + output_folder_name + '\\_cell-info.csv')
    return

def tabulate_cells(mask, n_cell):
    result = [[] for i in range(n_cell)]
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            try:
                cell = int(mask[x][y])
                result[cell - 1].append((x, y))
            except:
                continue
    return result

def get_file_names_with_ext(path: str, ext: str):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fname = file.split('.')
            if (fname[-1] == ext):
                flist.append(root + '\\' + file)
    return flist

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
