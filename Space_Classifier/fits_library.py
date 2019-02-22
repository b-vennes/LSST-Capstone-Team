import numpy as np
from astropy import wcs
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
import sys
import Images
import sys
import math
sys.path.append("..")
from Pipeline.Database_Connect import DynamoConnect

def parse_images(image_id_list):
    # Open file containing image ids and store them in a list
    sources = []
    with open(image_id_list) as f:
        lines = f.read().splitlines()
    for s_id in lines:
        sources.append(s_id)

    ALL_STARS = []
    ALL_NON_STARS = []

    SPACING = 128

    # Open fits files for every source and format stars and non-stars
    for fts in sources:
        hdulist = fits.open("Images/" + fts + ".fits")
        # Get data for stars and non-stars
        stars = DynamoConnect.get_stars(fts)
        non_stars = DynamoConnect.get_non_stars(fts)

        fits_data = hdulist[1].data

        print("processing", fts)

        # Each source has x and y WSC - take that and convert to pixel locations
        for star in stars:
            px = int(round(float(star.get('x')) - hdulist[1].header['CRVAL1A']))
            py = int(round(float(star.get('y')) - hdulist[1].header['CRVAL2A']))

            object_array = fits_data[(px-SPACING):(px+SPACING),(py-SPACING):(py+SPACING)]

            if object_array.shape != (256,256):
                continue

            ALL_STARS.append(object_array)

        for nstar in non_stars:
            px = int(round(float(nstar.get('x')) - hdulist[1].header['CRVAL1A']))
            py = int(round(float(nstar.get('y')) - hdulist[1].header['CRVAL2A']))

            object_array = fits_data[(px-SPACING):(px+SPACING),(py-SPACING):(py+SPACING)]

            if object_array.shape != (256,256):
                continue

            ALL_NON_STARS.append(object_array)

        hdulist.close()
            
    # Seperate 80% of true-positives and true-negatives for training
    # Seperate other 20% for validation
    num_star_training = int(len(ALL_STARS) * 0.8)
    num_star_validation = len(ALL_STARS) - num_star_training
    num_nstar_training = int(len(ALL_NON_STARS) * 0.8)
    num_nstar_validation = len(ALL_NON_STARS) - num_nstar_training
    
    # Create training and validation lists
    training_set = []
    training_labels = []
    validation_set = []
    validation_labels = []
    for i in range(0,num_star_training):
        training_set.append(ALL_STARS[i])
        training_labels.append(1)
    for i in range(0,num_nstar_training):
        training_set.append(ALL_NON_STARS[i])
        training_labels.append(0)
    for i in range(num_star_training, num_star_training + num_star_validation):
        validation_set.append(ALL_STARS[i])
        validation_labels.append(1)
    for i in range(num_nstar_training, num_nstar_training + num_nstar_validation):
        validation_set.append(ALL_NON_STARS[i])
        validation_labels.append(0)

    training_set = np.stack(training_set, axis=0)
    training_set = np.expand_dims(training_set, axis=3)

    validation_set = np.stack(validation_set, axis=0)
    validation_set = np.expand_dims(validation_set, axis=3)

    training_labels = np.stack(training_labels, axis=0)
    training_labels = np.expand_dims(training_labels, axis=1)

    validation_labels = np.stack(validation_labels, axis=0)
    validation_labels = np.expand_dims(validation_labels, axis=1)

    return training_set, training_labels, validation_set, validation_labels

