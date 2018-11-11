from bs4 import BeautifulSoup
import urllib.request
import re
import os
import random

def get_hubble_comets(url="http://hubblesite.org/images/gallery/46-comets"):
    # 80 : 20 rule 80% will be training data 20% will be validation data
    base_url = "http://hubblesite.org"
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page)
    for link in soup.findAll('a'):
        if link.get('href') is None:
            continue

        # print("image page links")
        if "/gallery/46-comets" in link.get('href') and "page" not in link.get('href'):
            # print(link.get('href'))
            scrape(base_url+link.get('href'))

        # print("next page links")
        if "/gallery/46-comets" in link.get('href') and "page" in link.get('href'):
            get_next_page(base_url+link.get('href'))
            print(link.get('href'))

def get_next_page(url):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page)
    for link in soup.findAll('a'):
        if link.get('href') is None:
            continue
        # print("image page links")
        if "/gallery/46-comets" in link.get('href') and "page" not in link.get('href'):
            # print(link.get('href'))
            scrape(base_url+link.get('href'))


def scrape(url):
    image_page = urllib.request.urlopen(url)
    image_soup = BeautifulSoup(image_page)
    for image_link in image_soup.findAll('a'):
        # print(image_link.get('href'))
        if "imgsrc" in image_link.get('href') and "jpg" in image_link.get('href'):
            urllib.request.urlretrieve(image_link.get('href'), "local.jpg")
            # TODO:need a function to put this in the bolb and the DB
            # if random.randint(0,10) > 2:
            #     # part of training set
            # else:
            #     # part of validation set
            # TODO:mark all the sources in the image
            os.reomve("local.jpg")
            break



def get_hubble_other():
    # get images that dont contain comets with the same 80 : 20 rule
    pass
