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
    col_bd_fixed = 'F:\\DiffusionAnalysis\\test1\\_ColBD_fixed-particles.csv'
    col_bd_spots = 'F:\\DiffusionAnalysis\\test1\\_ColBD_spots.csv'

    box_size = 4


    return

'''
================================================================================================================
START
================================================================================================================
'''

# Start Script
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))