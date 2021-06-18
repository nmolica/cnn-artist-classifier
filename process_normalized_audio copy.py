import os
import time
import numpy as np
import librosa
import librosa.display
import threading
import matplotlib.pyplot as plt
from PIL import Image
from pydub import AudioSegment

print("Please ensure that all audio files are in wav PCM 8-bit format with sampling rate of 8000kHz, and mono instead of multi-channel.")
print("Specify source folder for audio files:")
src = input()
print("Specify destination folder for training/testing data:")
dest = input()

if not os.path.isdir(src) or not os.path.isdir(dest):
    print("Both the source and the destination must be directories.")
    exit(1)

clipped_audio = dest + "/clipped_audio"
os.mkdir(clipped_audio)
plotted_clips = dest + "/plotted_clips"
os.mkdir(plotted_clips)

segment_size = 30 * 1000 # 30 second audio clips
left, right, top, bottom = 6, 779, 22, 76 # cropping indices for plotted audio

def split_audio(audio):
    """ Split up the audio into 30-second interleaving segments, where each
    consecutive segment jumps forward by 1/3 of the segment size.
    """
    segments = []
    top = 0
    bottom = segment_size
    while bottom < len(audio):
        segments.append(audio[top:bottom])
        top += segment_size / 3
        bottom += segment_size / 3
    return segments

def export_audio_segments(prefix, pieces):
    i = 0
    for p in pieces:
        p.export(clipped_audio + "/" + str(i + 1) + prefix, format='wav')
        i += 1

def plot_audio(path):
	y, sr = librosa.load(path)
	spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
	spect = librosa.power_to_db(spect, ref=np.max)
	plt.figure(figsize=(10,1))
	librosa.display.specshow(spect, fmax=8000)
	plt.savefig(plotted_clips + "/" + path.split("/")[-1].split(".")[0] + ".png", bbox_inches='tight', pad_inches=0)
	plt.close()

def crop(path):
    im = Image.open(plotted_clips + "/" + path)
    im.crop((left, top, right, bottom)).save(dest + "/" + path, quality = 100)

def segmenting_thread_func():
    for file in os.listdir(src):
        if file[-4:] == '.wav':
            export_audio_segments(file, split_audio(AudioSegment.from_wav(src + "/" + file)))
    print("Finished segmenting audio.")

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

segmenter = threading.Thread(target=segmenting_thread_func)
cropper = threading.Thread(target=cropping_thread_func)

segmenter.start()
cropper.start()

plotting_thread_func() # this must happen in the main thread

segmenter.join()
cropper.join()

os.rmdir(clipped_audio)
os.rmdir(plotted_clips)