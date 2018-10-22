import os
import boto3

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
