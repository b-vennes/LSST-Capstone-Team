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
                    print(type(cutout.image))
                except:
                    print("Out of bounds -- fix that man")


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

def post_database(image_id):

    # predefined values for region and endpoint
    dynamodb_region = 'us-west-2'
    dynamodb_endpoint ='http://dynamodb.us-west-2.amazonaws.com'

    # access the dynamodb
    dynamodb = boto3.resource('dynamodb', region_name=dynamodb_region, endpoint_url=dynamodb_endpoint)

    table_name = 'LSST-Images'

    table = dynamodb.Table(table_name)

    response = table.put_item(
        Item={
            'ImageId': image_id,
            'Label': 'NULL'
        }
    )

def get_random_image_url():

    # predefined values for region and endpoint
    dynamodb_region = 'us-west-2'
    dynamodb_endpoint ='http://dynamodb.us-west-2.amazonaws.com'

    # access the dynamodb
    dynamodb = boto3.resource('dynamodb', region_name=dynamodb_region, endpoint_url=dynamodb_endpoint)

    table_name = 'LSST-Images'

    table = dynamodb.Table(table_name)

    response = table.scan()

    print(response['Count'])

    randomIndex = random.randint(0,response['Count'] - 1)

    items = response['Items']

    random_image_id = items[randomIndex]['ImageId']

    link = get_image_link(random_image_id)

    print(link)

    return link

def get_image_link(image_id):

    # our bucket name
    bucket_name = 'lsst-images'

    # get the zone for the bucket, to be added to the url
    bucket_zone = boto3.client('s3').get_bucket_location(Bucket=bucket_name)

    image_link = "https://s3-{0}.amazonaws.com/{1}/{2}.png".format(
        bucket_zone['LocationConstraint'],
        bucket_name,
        image_id
    )

    return image_link

# get_random_image_url()
create_image()
