## Run

1. Download dataset from [Kaggle](https://www.kaggle.com/c/silero-audio-classifier/data) and place it inside repository directory.

2. Launch script for converting wav to jpg (by default it's using all your cpu threads, but you can change it inside code).
```bash
python wav2jpg.py
```
After this you got my_dataset.csv with jpg filename, target. Now you need to split it  to train.csv and test.csv


3.1 For training CNN 
```
python train.py
```
3.2 Make prediction for submission
```
python inference.py
```
4 Clusterization binary images with k-means. Got only 0.415. I think, we need to reduce number of features, here i have 1d array with size 128x72 = 9216. Maybe we can apply here linear transformation, just like in neural networks.
```
python kmeans_images.py
```
