import numpy as np
from astropy import wcs
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
import sys
import Images
import sys
sys.path.append("..")
from Pipeline.Database_Connect import DynamoConnect

NUMBER_LARGE_IMAGES = 0
NUMBER_CROPPED_IMAGES = 0

def parse_images(image_id_list):
    # Open file containing image ids and store them in a list
    sources = []
    with open(image_id_list) as f:
        lines = f.read().splitlines()
    for s_id in lines:
        print(s_id)
        sources.append(s_id)

    # Open fits files for every source and collect the headers
    for fts in sources:
        hdulist = fits.open("Images/" + fts + ".fits")
        # Parse the WCS keywords in the primary HDU
        w = wcs.WCS(hdulist[1].header)
        print(w)

    # Get data for stars and non-stars
    stars = DynamoConnect.get_stars()
    non_stars = DynamoConnect.get_non_stars()
    print(stars[0])

    # TO-DO: 
    # Each source has x and y - take that and convert to actual locations
##    for star in stars:
##        px = star.x - CRVAL1A
##        py = star.y - CRVAL2A
##        add px and py to star dictionary

    # 14 pixels on both sides
    # seperate into training and testing

parse_images("Images/image_ids.list")
