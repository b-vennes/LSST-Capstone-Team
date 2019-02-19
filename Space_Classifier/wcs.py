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
        sources.append(s_id)

    # Open fits files for every source and format stars and non-stars
    ALL_STARS = []
    ALL_NON_STARS = []
    for fts in sources:
        hdulist = fits.open("Images/" + fts + ".fits")

        # Get data for stars and non-stars
        stars = DynamoConnect.get_stars(fts)
        non_stars = DynamoConnect.get_non_stars(fts)

        # Each source has x and y WSC - take that and convert to pixel locations
        for star in stars:
            px = float(star.get('x')) - hdulist[1].header['CRVAL1A']
            py = float(star.get('y')) - hdulist[1].header['CRVAL2A']
            star['px'] = px
            star['py'] = py
            ALL_STARS.append(star)
        for nstar in non_stars:
            px = float(nstar.get('x')) - hdulist[1].header['CRVAL1A']
            py = float(nstar.get('y')) - hdulist[1].header['CRVAL2A']
            nstar['px'] = px
            nstar['py'] = py
            ALL_NON_STARS.append(nstar)
            
##    print(ALL_STARS[0])
##    print(ALL_STARS[1])
##    print(ALL_NON_STARS[0])
##    print(ALL_NON_STARS[1])

    # TO-DO:
    # 14 pixels on both sides
    # seperate into training and testing

parse_images("Images/image_ids.list")
