""" This script takes a folder full of color png photos and converts it to grayscale.
"""

import os
from PIL import Image

# get source and destination folders from the user so we know where to take photos from
# and where to put the finished ones
print("Source folder:")
src = input()
print("Destination folder:")
dest = input()

# validate the user-provided paths
if not os.path.isdir(src):
    print("Invalid source folder.")
    exit(1)
elif not os.path.isdir(dest):
    print("Invalid destination folder.")
    exit(1)

# loop through the files in the provided directory, convert them to grayscale, and save
# them to the destination directory
for file in os.listdir(src):
    if file[-4:] == '.png':
        Image.open(src + "/" + file).convert('LA').save(dest + "/" + file)

print("Successfully converted images to black and white.")