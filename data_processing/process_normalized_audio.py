""" This script handles processing audio to prep it for training. It takes (normalized) songs and
segments, plots and crops them. Note that it is required for the audio to be in mono wav PCM 8-bit
format with a sampling rate of 8000kHz before processing can begin. This script is multithreaded
for the sake of efficiency. Regretably, the matplotlib library requires all plotting to happen in
the main thread, so this is a considerable bottleneck that causes the code to require several
hours in order to process any meaningful amount of data.
"""

import os
import time
import numpy as np
import librosa
import librosa.display
import threading
import matplotlib.pyplot as plt
from PIL import Image
from pydub import AudioSegment

# ensure that the user has already converted all audio files to the proper format
print("Please ensure that all audio files are in wav PCM 8-bit format with sampling rate of 8000kHz, and mono instead of multi-channel.")
print("Specify source folder for audio files:")
src = input()
print("Specify destination folder for training/testing data:")
dest = input()

# verify the validity of user-provided paths
if not os.path.isdir(src) or not os.path.isdir(dest):
    print("Both the source and the destination must be directories.")
    exit(1)

# create staging directories and define useful global vars
clipped_audio = dest + "/clipped_audio"
os.mkdir(clipped_audio)
plotted_clips = dest + "/plotted_clips"
os.mkdir(plotted_clips)
segment_size = 30 * 1000 # 30 second audio clips
left, right, top, bottom = 6, 779, 22, 76 # cropping indices for plotted audio

# Split up the audio into 30-second interleaving segments, where each
# consecutive segment jumps forward by 1/3 of the segment size.
def split_audio(audio):
    segments = []
    top = 0
    bottom = segment_size
    while bottom < len(audio):
        segments.append(audio[top:bottom])
        top += segment_size / 3
        bottom += segment_size / 3
    return segments

# Save the provided audio segments to the staging area dedicated to segmented audio.
def export_audio_segments(prefix, pieces):
    i = 0
    for p in pieces:
        p.export(clipped_audio + "/" + str(i + 1) + prefix, format='wav')
        i += 1

# Load the audio segment at a provided path and plot it as a mel spectrogram.
def plot_audio(path):
	y, sr = librosa.load(path)
	spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
	spect = librosa.power_to_db(spect, ref=np.max)
	plt.figure(figsize=(10,1))
	librosa.display.specshow(spect, fmax=8000)
	plt.savefig(plotted_clips + "/" + path.split("/")[-1].split(".")[0] + ".png", bbox_inches='tight', pad_inches=0)
	plt.close()

# Crop a mel spectrogram to remove unnecessary borders, ensuring that the model
# only sees the information that is necessary.
def crop(path):
    im = Image.open(plotted_clips + "/" + path)
    im.crop((left, top, right, bottom)).save(dest + "/" + path, quality = 100)

# The target function for the thread that segments audio.
def segmenting_thread_func():
    for file in os.listdir(src):
        if file[-4:] == '.wav':
            export_audio_segments(file, split_audio(AudioSegment.from_wav(src + "/" + file)))
    print("Finished segmenting audio.")

# The target function for the thread that plots audio as mel spectrograms.
def plotting_thread_func():
    while True:
        for file in os.listdir(clipped_audio):
            if file[-4:] == '.wav':
                plot_audio(clipped_audio + "/" + file)
                os.remove(clipped_audio + "/" + file)

        # pause for 2 seconds in case we're waiting on another thread, break if more files don't appear
        time.sleep(2)
        try:
            os.remove(clipped_audio + "/" + ".DS_Store")
        except FileNotFoundError:
            pass
        if len(os.listdir(clipped_audio)) == 0:
            break
    print("Finished plotting audio.")

# The target function for the thread that crops mel spectrograms.
def cropping_thread_func():
    while True:
        for file in os.listdir(plotted_clips):
            if file[-4:] == '.png':
                crop(file)
                os.remove(plotted_clips + "/" + file)

        # pause for 5 seconds in case we're waiting on another thread, break if more files don't appear
        time.sleep(5)
        try:
            os.remove(plotted_clips + "/" + ".DS_Store")
        except FileNotFoundError:
            pass
        if len(os.listdir(plotted_clips)) == 0:
            break
    print("Finished cropping images.")

# initialize the segmenting and cropping threads
segmenter = threading.Thread(target=segmenting_thread_func)
cropper = threading.Thread(target=cropping_thread_func)

# start the segmenting and cropping threads
segmenter.start()
cropper.start()

# do all plotting in main thread (this is because the matplotlib library requires all plotting
# to happen in the main thread or it will crash)
plotting_thread_func()

# wait for the segmenting and cropping threads to finish their tasks
segmenter.join()
cropper.join()

# remove the staging directories
os.rmdir(clipped_audio)
os.rmdir(plotted_clips)