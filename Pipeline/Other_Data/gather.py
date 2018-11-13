from bs4 import BeautifulSoup
import urllib.request
import re
import os
import random
import DynamoConnect as db

def get_hubble_data(url, gallery_name, label):
    # 80 : 20 rule 80% will be training data 20% will be validation data
    base_url = "http://hubblesite.org"
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, features="html.parser")
    pages = []
    for link in soup.findAll('a'):
        if link.get('href') is None:
            continue

        # print("image page links")
        if "/gallery/"+gallery_name in link.get('href') and "page" not in link.get('href'):
            # print(link.get('href'))
            scrape(base_url+link.get('href'), gallery_name, label)

        # print("next page links")
        if "/gallery/"+gallery_name in link.get('href') and "page" in link.get('href'):
            pages.append(base_url+link.get('href'))
    pages = list(set(pages))
    for page in pages:
        print(page)
        get_next_page(page, gallery_name, label)

def get_next_page(url, gallery_name, label):
    base_url = "http://hubblesite.org"
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, features="html.parser")
    for link in soup.findAll('a'):
        if link.get('href') is None:
            continue
        # print("image page links")
        if "/gallery/"+gallery_name in link.get('href') and "page" not in link.get('href'):
            # print(link.get('href'))
            scrape(base_url+link.get('href'), gallery_name, label)


def scrape(url, gallery_name, label):
    image_page = urllib.request.urlopen(url)
    image_soup = BeautifulSoup(image_page, features="html.parser")
    for image_link in image_soup.findAll('a'):
        # print(image_link.get('href'))
        if image_link.get('href') is None:
            continue
        if "imgsrc" in image_link.get('href') and "jpg" in image_link.get('href'):
            urllib.request.urlretrieve(image_link.get('href'), "local.jpg")
            # TODO:mark all the sources in the image
            # TODO:need a function to put this in the bolb and the DB
            if random.randint(0,10) > 2:
                #part of training set
                link = db.upload_image("local.jpg", url, label, False)
                print(link)
            else:
                # part of validation set
                link = db.upload_image("local.jpg", url, label, True)
                print(link)
            break



def get_hubble_other(url="http://hubblesite.org/images/gallery/2-stars"):
    # get images that dont contain comets with the same 80 : 20 rule
    pass
