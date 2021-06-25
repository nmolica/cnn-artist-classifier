""" This script takes a folder of songs from the 15 artists and partitions a single
song from each artist to serve as test data. This script is not generalized, so it
will not work with any artist at all and it requires the file names to be in a
specific format.
"""

import os
import random

# get source and destination directories from the user
print("Source folder for data:")
src = input()
print("Destination fold for test data:")
dest = input()

# validate the user-provided paths
if not os.path.isdir(src):
    print("Invalid source folder.")
elif not os.path.isdir(dest):
    print("Invalid destination folder.")

# initialize a dictionary of empty sets to map artist names to their songs
songs = {
    'chetbaker': set(),
    'billevans': set(),
    'johncoltrane': set(),
    'mccoytyner': set(),
    'bach': set(),
    'mumfordandsons': set(),
    'gregoryalanisakov': set(),
    'mandolinorange': set(),
    'thesteeldrivers': set(),
    'bts': set(),
    'chopin': set(),
    'mamamoo': set(),
    'mozart': set(),
    'seventeen': set(),
    'tchaikovsky': set()
}

# Given a file name, return the name of the artist who wrote it. Note that
# this requires the file name to be in our specific format. You can look at
# the data files in bw_test and bw_train to see what this is.
def get_artist(file_name):
    segment_and_artist = file_name.split("_")[0]
    if segment_and_artist[1:] in songs:
        artist = segment_and_artist[1:]
    elif segment_and_artist[2:] in songs:
        artist = segment_and_artist[2:]
    elif segment_and_artist[3:] in songs:
        artist = segment_and_artist[3:]
    else:
        print("Invalid file name scheme.")
        exit(1)

    return artist

# loop through all files and add the name of the file to the set in the songs dictionary
for file in os.listdir(src):
    if file != ".DS_Store":
        songs[get_artist(file)].add(file)

# for each artist, partition all files associated with one randomly selected song of theirs
test_partition = []
for artist in songs:
    test_partition.append(random.choice(list(songs[artist])).split("_")[1].split(".")[0])

# move all the selected test files to the user-provided test data directory, deleting
# them from the training data in the process
for file in os.listdir(src):
    if file.split("_")[1].split(".")[0] in test_partition:
        os.rename(src + "/" + file, dest + "/" + file)

print("Successfully partition test data.")