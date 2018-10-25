import os
import boto3
import subprocess
import lsst.daf.persistence as dafPersist
import lsst.afw.display as afwDisplay

def create_image():
    butler = dafPersist.Butler(inputs='DATA/rerun/processCcdOutputs')
    display = afwDisplay.getDisplay(backend='ds9')
    for data in butler.queryMetadata('calexp', ['visit', 'ccd'], dataId={'filter': 'HSC-R'}):
        dataId = {'filter': 'HSC-R', 'visit': data[0], 'ccd': data[1]}
        calexp = butler.get('calexp', **dataId)
        display.mtv(calexp)
        #mark all sources
        src = butler.get('src', dataId={'filter': 'HSC-R', 'visit': data[0], 'ccd': data[1]})
        with display.Buffering():
            for s in src:
                display.dot("o", s.getX(), s.getY(), size=10, ctype='orange')
        #crop each source

        #put each croped image in blob

    return 0

def post_image(image_path):
    """
    Posts the given image to the amazon s3 lsst-images bucket

    returns the link to the image on s3
    """

    # our bucket name
    bucket_name = 'lsst-images'

    # get amazon storage
    s3 = boto3.resource('s3')

    image = open(image_path, 'rb')
    s3.Bucket(bucket_name).put_object(Key=image_path, Body=image)

    # get the zone for the bucket, to be added to the url
    bucket_zone = boto3.client('s3').get_bucket_location(Bucket=bucket_name)

    # get the link for the image, this will be sent to the database
    image_link = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
        bucket_zone['LocationConstraint'],
        bucket_name,
        image_path
    )

    return image_link

post_image('comet.jpg')
