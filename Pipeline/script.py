import os
import boto3
import subprocess
import random
import uuid
import urllib.request
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from shutil import copyfile
import lsst.daf.persistence as dafPersist
import lsst.afw.display as afwDisplay


# predefined values for region and endpoint
dynamodb_region = 'us-west-2'
dynamodb_endpoint ='http://dynamodb.us-west-2.amazonaws.com'

def upload_image(image_path):
    """
    Uploads the given image to the s3 bucket with a random identifier.
    Creates a new item in the database with the identifier.
    """

    # generate a random unique string, highly unlikely that two of these will clash
    unique_id = str(uuid.uuid4())

    # make a copy of the image to a new file with the unique file path
    copy_image = unique_id + '.fits'
    copyfile(image_path, copy_image)

    # send new unique file to s3 and delete it from local storage
    new_link = post_image(copy_image)
    os.remove(copy_image)

    # add the unique id to the database
    post_database(unique_id)

    return new_link

###############################################################
################################################################

def post_image(image_path):
    """
    Posts the given image to the amazon s3 lsst-images bucket.
    Returns the link to the image on s3.
    """

    # our bucket name
    bucket_name = 'lsst-patches'

    # get amazon storage
    s3 = boto3.resource('s3')

    # upload image to s3 and make it public readable
    s3.Bucket(bucket_name).upload_file(image_path, image_path, ExtraArgs={'ACL':'public-read'})

    # get the zone for the bucket, to be added to the url
    bucket_zone = boto3.client('s3').get_bucket_location(Bucket=bucket_name)

    # get the link for the image, this will be sent to the database
    image_link = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
        bucket_zone['LocationConstraint'],
        bucket_name,
        image_path
    )

    return image_link

###############################################################
################################################################

def post_image_database(image_id):
    """
    Creates a new item in the database with the given image id.
    """

    # access the dynamodb
    dynamodb = boto3.resource('dynamodb', region_name=dynamodb_region, endpoint_url=dynamodb_endpoint)

    images_table = dynamodb.Table('lsst-images')

    put_response = images_table.put_item(
        Item={
            'id': image_id,
        }
    )

    return put_response

###############################################################
################################################################

def post_src_database(image_id, x, y, label):
    """
    Creates a new item in the database with the given image id.
    """
    unique_id = str(uuid.uuid4())

    # access the dynamodb
    dynamodb = boto3.resource('dynamodb', region_name=dynamodb_region, endpoint_url=dynamodb_endpoint)

    images_table = dynamodb.Table('lsst-image-src')

    put_response = images_table.put_item(
        Item={
            'id': unique_id,
            'image_id' : image_id,
            'x':x,
            'y':y,
            'label': label,
        }
    )

    return put_response

###############################################################
################################################################

def post_top_src_database(id, image_id, x, y, label):
    """
    Creates a new item in the database with the given image id.
    """
    # unique_id = str(uuid.uuid4())

    # access the dynamodb
    dynamodb = boto3.resource('dynamodb', region_name=dynamodb_region, endpoint_url=dynamodb_endpoint)

    images_table = dynamodb.Table('lsst-top-src')

    put_response = images_table.put_item(
        Item={
            'id': id,
            'image_id' : image_id,
            'x':x,
            'y':y,
            'label': label,
        }
    )

    return put_response

###############################################################
################################################################

def test():
    butler = dafPersist.Butler(inputs='DATA/rerun/coaddForcedPhot')

    coadd = butler.get( "deepCoadd_calexp", tract=0, patch='1,1', filter='HSC-I')

    rSources = butler.get('deepCoadd_forced_src', {'filter': 'HSC-R', 'tract': 0, 'patch': '1,1'})
    iSources = butler.get('deepCoadd_forced_src', {'filter': 'HSC-I', 'tract': 0, 'patch': '1,1'})

    rCoaddCalexp = butler.get('deepCoadd_calexp',  {'filter': 'HSC-R', 'tract': 0, 'patch': '1,1'})
    rCoaddCalib = rCoaddCalexp.getCalib()
    iCoaddCalexp = butler.get('deepCoadd_calexp',  {'filter': 'HSC-I', 'tract': 0, 'patch': '1,1'})
    iCoaddCalib = iCoaddCalexp.getCalib()

    ##############################################
    # unique_id = str(uuid.uuid4())
    # iCoaddCalib.writeFits(unique_id+".fits")
    # file_name = "patch_1_1"+".fits"
    # coadd.writeFits(file_name)
    # new_link = post_image(file_name)
    # os.remove(file_name)
    display = afwDisplay.getDisplay()
    display.mtv(coadd)
    display.setMaskTransparency(100)
    display.scale("asinh", -1, 30)
    ###################################################

    rCoaddCalib.setThrowOnNegativeFlux(False)
    iCoaddCalib.setThrowOnNegativeFlux(False)

    # print(iSources.getSchema())

    # rMags = rCoaddCalib.getMagnitude(rSources['base_PsfFlux_instFlux'])
    iMags = iCoaddCalib.getMagnitude(iSources['base_PsfFlux_flux'])
    maxMag = max(iMags)

    isDeblended = rSources['deblend_nChild'] == 0

    refTable = butler.get('deepCoadd_ref', {'filter': 'HSC-R^HSC-I', 'tract': 0, 'patch': '1,1'})

    inInnerRegions = refTable['detect_isPatchInner'] & refTable['detect_isTractInner']

    isSkyObject = refTable['merge_peak_sky']

    isPrimary = refTable['detect_isPrimary']

    #rMags[isPrimary]
    #iMags[isPrimary]

    isStellar = iSources['base_ClassificationExtendedness_value'] < 1.

    isNotStellar = iSources['base_ClassificationExtendedness_value'] >= 1.

    isGoodFlux = ~iSources['base_PsfFlux_flag']

    selected = isPrimary & isStellar & isGoodFlux

    for src in iSources[selected]:
        # iMag = iCoaddCalib.getMagnitude(src['base_PsfFlux_flux'])
        # print(iMag)
        # if src.getX() > 5900 and src.getX() < 6000 and src.getY() > 6000 and src.getY() < 6100:
        #     print(str(src.getX())+ ", "+str(src.getY()))
        #     display.dot("o", src.getX(), src.getY(), size=10, ctype='orange')
        #
        # if iMag < maxMag/2.75:
        #     print(str(src.getX())+ ", "+str(src.getY()))
        display.dot("o", src.getX(), src.getY(), size=20, ctype='green')

###############################################################
################################################################

def main():

    butler = dafPersist.Butler(inputs='DATA/rerun/coaddForcedPhot')

    patches = ['0,1', '1,0', '1,1', '1,2', '2,0', '2,1', '2,2']

    id = 0

    for patch in patches:
        coadd = butler.get( "deepCoadd_calexp", tract=0, patch=patch, filter='HSC-I')

        rSources = butler.get('deepCoadd_forced_src', {'filter': 'HSC-R', 'tract': 0, 'patch': patch})
        iSources = butler.get('deepCoadd_forced_src', {'filter': 'HSC-I', 'tract': 0, 'patch': patch})

        rCoaddCalexp = butler.get('deepCoadd_calexp',  {'filter': 'HSC-R', 'tract': 0, 'patch': patch})
        rCoaddCalib = rCoaddCalexp.getCalib()
        iCoaddCalexp = butler.get('deepCoadd_calexp',  {'filter': 'HSC-I', 'tract': 0, 'patch': patch})
        iCoaddCalib = iCoaddCalexp.getCalib()

        ###############################################
        result = [x.strip() for x in patch.split(',')]
        file_name = "patch_" + result[0] + "_" + result[1] + ".fits"
        coadd.writeFits(file_name)

        # send new unique file to s3 and delete it from local storage
        new_link = post_image(file_name)
        # os.remove(file_name)

        # post_image_database(unique_id)
        ##################################################

        rCoaddCalib.setThrowOnNegativeFlux(False)
        iCoaddCalib.setThrowOnNegativeFlux(False)

        iMags = iCoaddCalib.getMagnitude(iSources['base_PsfFlux_flux'])
        maxMag = max(iMags)

        isDeblended = rSources['deblend_nChild'] == 0

        refTable = butler.get('deepCoadd_ref', {'filter': 'HSC-R^HSC-I', 'tract': 0, 'patch': patch})

        inInnerRegions = refTable['detect_isPatchInner'] & refTable['detect_isTractInner']

        isSkyObject = refTable['merge_peak_sky']

        isPrimary = refTable['detect_isPrimary']


        isStellar = iSources['base_ClassificationExtendedness_value'] < 1.

        isNotStellar = iSources['base_ClassificationExtendedness_value'] >= 1.

        isGoodFlux = ~iSources['base_PsfFlux_flag']

        selected = isPrimary & isNotStellar & isGoodFlux



        for src in iSources[selected]:
            iMag = iCoaddCalib.getMagnitude(src['base_PsfFlux_flux'])
            # print(str(src.getX())+ ", "+str(src.getY()))
            if iMag < maxMag/2.75:
                post_top_src_database(str(id), file_name, str(src.getX()), str(src.getY()), 'EXT')
                id += 1

        selected2 = isPrimary & isStellar & isGoodFlux

        for src in iSources[selected2]:
            iMag = iCoaddCalib.getMagnitude(src['base_PsfFlux_flux'])
            # print(str(src.getX())+ ", "+str(src.getY()))
            if iMag < maxMag/2.75:
                post_top_src_database(str(id), file_name, str(src.getX()), str(src.getY()), 'STAR')
                id += 1

        print(id)
