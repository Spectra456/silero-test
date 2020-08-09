import matplotlib.pyplot as plt
import librosa.display
import pandas as pd 
import numpy as np
import librosa
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

def wav2jpg(id): #filename, target):
	filename = np_array[id][0]
	y, sr = librosa.load(filename)
	y = y[:100000] # shorten audio a bit for speed
	window_size = 1024
	window = np.hanning(window_size)
	stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
	out = 2 * np.abs(stft) / np.sum(window)
	fig = plt.Figure()
	canvas = FigureCanvas(fig)
	ax = fig.add_subplot(111)
	p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)
	save_filepath = 'dataset_jpg/{}.jpg'.format(Path(filename).stem)
	fig.savefig(save_filepath,bbox_inches='tight', pad_inches=0)

df = pd.read_csv('train.csv')
df = df.drop(columns=[df.columns[0], 'label'])

np_array = df.to_numpy()
dataset_array = df.to_numpy()
for i in range(len(dataset_array)):
	dataset_array[i][0] = '{}.jpg'.format(Path(np_array[i][0]).stem)

print(dataset_array)
pd.DataFrame(dataset_array).to_csv("my_dataset.csv")

pool = Pool(8)
list(tqdm(pool.imap_unordered(wav2jpg, range(0,len(np_array))), total=len(np_array)))