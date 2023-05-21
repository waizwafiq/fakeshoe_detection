from bs4 import BeautifulSoup
import cv2
import requests 
import os


URL = "https://realpython.github.io/fake-jobs/"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
print(str(soup))

def saveAsFile(filename, value, folder='saved', extension='.txt'):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    filename = f'./{folder}/{filename}{extension}'
    file = open(filename, 'w')

    file.write(str(value))

    print(f"Saved {filename} as {extension}!")
    file.close()

def saveImage(filename, img, folder='saved_img', extension='.png'):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    filename = f"./{folder}/{filename}{extension}" 
    cv2.imwrite(filename, img)

    print(f"Saved {filename} as an image!")

