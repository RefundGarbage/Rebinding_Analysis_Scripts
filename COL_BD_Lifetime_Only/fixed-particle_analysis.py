import numpy as np
import pandas as pd
import time
import os
import logging
import csv
import shutil
from natsort import natsorted
from skimage import io as imgio


def main():
    csv_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\Tracking'  # csv from trackmate
    particle_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\particles_result'
    mask_path = 'F:\\MicroscopyTest\\20231210_Dataset\\Fixed_particle\\wt\\_seg'

    # Some parameters
    dist_bound_particle = 1.5

    output_path = csv_path + '\\_ColBD_LIFE'
    logging_setup(output_path, 'fixed-particle_analysis')

    rebind_events_strict = pd.read_csv(output_path + '\\_ColBD_LIFE_rebind-strict.csv')
    rebind_events_strict = rebind_events_strict.loc[:, ~rebind_events_strict.columns.str.contains('^Unnamed')]

    particle_files = natsorted(get_file_names_with_ext(particle_path, 'csv'))
    particle_videos = []
    for file in particle_files:
        particles = pd.read_csv(file)
        particle_videos.append(particles[
                                   ['X_(px)', 'Y_(px)', 'xMin', 'yMin', 'xMax', 'yMax']
                                ])

    mask_files = natsorted(get_file_names_with_ext(mask_path, 'png'))
    mask_videos = []
    for file in mask_files:
        mask_videos.append(np.swapaxes(imgio.imread(file), 0, 1))

    if not len(mask_videos) == len(particle_videos):
        raise ValueError

    count_event_no_particle = 0
    rebinds_particle_result = []

    rebinds_videos = split_rebinds_by_video(rebind_events_strict)
    for i in range(len(mask_videos)):
        print_log('Processing:', mask_files[i])
        n_cell = np.max(mask_videos[i])
        print_log('\t# Cells in Video:', n_cell)
        particle_cells = split_particles_by_cell(particle_videos[i], mask_videos[i], n_cell)
        rebinds_cells = split_rebinds_by_cell(rebinds_videos[i], n_cell)

        for j in range(len(particle_cells)):
            print_log('\t-> Cell', j, end='')
            if len(rebinds_cells[j]) == 0:
                print_log(' [ NO REBINDING EVENT ]')
                continue

            rebinds = rebinds_cells[j]
            header = list(rebinds[['Video #', 'Cell']].to_numpy()[0].flatten().astype('int'))

            if len(particle_cells[j]) == 0:
                _ = 0
                for index, row in rebinds.iterrows():
                    rebinds_particle_result.append(
                        header + [index+1, -1, -1, row['Time'], row['Speed'], row['Distance'], float('inf'), float('inf')]
                    )
                    _ += 1
                print_log(' [ NO PARTICLE IN CELL,', _, 'EVENTS SET TO -1 ]')
                count_event_no_particle += _
                continue

            particles = particle_cells[j]

            print_log(' [ REBINDING EVENTS WITH PARTICLES', len(rebinds), 'REBINDS', len(particles), 'PARTICLES ]')
            for index, row in rebinds.iterrows():
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                distances = []
                for particle in particles:
                    distances.append([distance([x1, y1], particle), distance([x2, y2], particle)])
                ind1, min1 = -1, float('inf')
                ind2, min2 = -1, float('inf')
                for k in range(len(distances)):
                    if(distances[k][0] < min1):
                        ind1 = k
                        min1 = distances[k][0]
                    if(distances[k][1] < min2):
                        ind2 = k
                        min2 = distances[k][1]
                decision1 = ind1 + 1 if min1 < dist_bound_particle else 0
                decision2 = ind2 + 1 if min2 < dist_bound_particle else 0
                print_log('\t\t-> Event', index+1)
                print_log('\t\t  : EVENT START minimum distance to particle:', min1, 'DECISION:', decision1)
                print_log('\t\t  : EVENT END minimum distance to particle:', min2, 'DECISION:', decision2)
                rebinds_particle_result.append(
                    header + [index+1, decision1, decision2, row['Time'], row['Speed'], row['Distance'], min1, min2]
                )
                continue
        continue
    print_log('# EVENTS set to -1 BY absent of particles:', count_event_no_particle)
    return

def distance(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))

def split_particles_by_cell(particles, mask, max_cell):
    result = [[] for i in range(max_cell)]
    coords = particles[['X_(px)', 'Y_(px)']].to_numpy()
    for i in range(coords.shape[0]):
        x, y = list(coords[i])
        try:
            cell = mask[int(x)][int(y)] - 1
            if cell >= 0: result[cell].append(particles.iloc[i].to_numpy())
        except:
            continue
    return result

def split_rebinds_by_video(rebinds):
    video_indices = rebinds[['Video #']].to_numpy().flatten()
    result = [[] for i in range(np.max(video_indices))]
    for i in range(video_indices.shape[0]):
        result[video_indices[i] - 1].append(rebinds.iloc[i].to_numpy())
    for i in range(len(result)):
        if len(result[i]) > 0:
            result[i] = pd.DataFrame(np.array(result[i]), columns=rebinds.columns)
    return result

def split_rebinds_by_cell(rebinds, n_cell):
    result = [[] for i in range(n_cell)]
    cell_indices = rebinds[['Cell']].to_numpy().flatten().astype('int')
    for i in range(cell_indices.shape[0]):
        result[cell_indices[i] - 1].append(rebinds.iloc[i].to_numpy())
    for i in range(len(result)):
        if len(result[i]) > 0:
            result[i] = pd.DataFrame(np.array(result[i]), columns=rebinds.columns)
    return result

def get_file_names_with_ext(path: str, ext: str):
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fname = file.split('.')
            if (fname[-1] == ext):
                flist.append(root + '\\' + file)
    return flist


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
