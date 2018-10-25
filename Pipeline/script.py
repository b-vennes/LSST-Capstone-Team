import os
import boto3
import subprocess
import random
from boto3.dynamodb.conditions import Key, Attr

def create_image():
    #data from butler:
    #   what data?
    print("start")
    subprocess.call("source bash_scripts/setup.bash")
    print("end")

    #Calibrating single frame
    # processCcd.py DATA --rerun processCcdOutputs --id
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

get_random_image_url()