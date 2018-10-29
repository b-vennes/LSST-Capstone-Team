import os
import boto3
import subprocess
import random
from boto3.dynamodb.conditions import Key, Attr
import lsst.daf.persistence as dafPersist
import lsst.afw.display as afwDisplay
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage


def create_image():
    butler = dafPersist.Butler(inputs='DATA/rerun/processCcdOutputs')
    display = afwDisplay.getDisplay(backend='ds9')
    for data in butler.queryMetadata('calexp', ['visit', 'ccd'], dataId={'filter': 'HSC-R'}):
        dataId = {'filter': 'HSC-R', 'visit': data[0], 'ccd': data[1]}
        calexp = butler.get('calexp', **dataId)
        #display.mtv(calexp)
        #mark all sources
        src = butler.get('src', dataId={'filter': 'HSC-R', 'visit': data[0], 'ccd': data[1]})
        with display.Buffering():
            for s in src:
                src_x = s.getX()
                src_y = s.getY()
                display.dot("o", src_x, src_y, size=10, ctype='orange')
                #crop each source
                try:
                    bbox = afwGeom.Box2I()
                    dim = calexp.getDimensions()
                    left = src_x - 300
                    right = src_x + 300
                    up = src_y - 300
                    down = src_y + 300
                    if src_x < 300:
                        left = 0
                        right = abs(src_x-300)
                    else:
                        left = src_x-300
                    if src_y < 300:
                        up = 0
                        down = abs(src_y-300)
                    if src_x + 300 > dim[0]:
                        right = dim[0]
                        left = 300 - (dim[0] - src_x)
                    if src_y + 300 > dim[1]:
                        down = dim[0]
                        up = 300 - (dim[1] - src_y)
                    bbox.include(afwGeom.Point2I(left,up))
                    bbox.include(afwGeom.Point2I(right,down))
                    cutout = calexp[bbox]
                    print(type(cutout))
                    try:
                        cutout.writeFits("test.fits")
                        print("wrote file?")
                        break
                    except:
                        print("write fits error")
                except:
                    print("Out of bounds -- fix that man")


        #put each croped image in blob

    return 0
