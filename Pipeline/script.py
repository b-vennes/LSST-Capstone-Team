import os
import requests
import subprocess

def create_image():
    #data from butler:
    #   what data?
    print("start")
    subprocess.call("source bash_scripts/setup.bash")
    print("end")

    #Calibrating single frame
    # processCcd.py DATA --rerun processCcdOutputs --id
    return 0

def post_images(image_path):

    # put image file into byte array
    with open(image_path, "rb") as imageFile:
        f = imageFile.read()
        b = bytearray(f)

    # make request
    data = requests.post("http://localhost:7071/api/lsst-label-api", data = b)

    # delete file after it has been sent
    os.remove(image_path)

    # check status code
    if (data.status_code == 200):
        return 1
    else:
        return 0

create_image()
