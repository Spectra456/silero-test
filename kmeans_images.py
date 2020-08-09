import cv2
import time
import glob
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
from pathlib import Path
import pickle
from sklearn.cluster import MiniBatchKMeans

images_path = 'submission_jpg/'

training_dataset = []

def read_image(id):
	filename = Path(list_files[id]).stem
	filename = images_path + filename + '.jpg'

	img = cv2.imread(filename)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_,img = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	img = cv2.normalize(img,  img, 0, 1, cv2.NORM_MINMAX)
	img = cv2.resize(img, (128,72))
	return img.flatten()

def inference(filename, kmeans_model):
	filename = Path(filename).stem
	filename = images_path + filename + '.jpg'
	img = cv2.imread(filename)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_,img = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	img = cv2.normalize(img,  img, 0, 1, cv2.NORM_MINMAX)
	img = cv2.resize(img, (128,72))
	return kmeans_model.predict(img.reshape(1, -1))


df = pd.read_csv('sample_submission.csv')
list_files = df['wav_path'].to_list()
pool = Pool(20)

for i in tqdm(pool.imap_unordered(read_image, range(0,len(list_files))),total=len(list_files)):
    training_dataset.append(i)

kmeans = MiniBatchKMeans(n_clusters=3, batch_size=64, verbose=2).fit(training_dataset)

df = pd.read_csv('sample_submission.csv')
submission_list = df['wav_path'].tolist()

sub_inf_list = []
for i in tqdm(range(len(submission_list))):
	result = inference(submission_list[i], kmeans)
	sub_inf_list.append([submission_list[i], result])
pd.DataFrame(sub_inf_list).to_csv('sub_kmeans.csv',index=False)