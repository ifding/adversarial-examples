
# Traffic Sign Classifier


### US LISA Dataset

The code in this repository is helpful to Convert the LISA Traffic Sign dataset into Tensorflow tfrecords.

1. Download LISA Dataset here : http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html

2. But only uses 17 classes in this project, as shown in `categories.txt` 


### German gtsrb Dataset


Download the dataset. You can download the pickled dataset in which we've already resized the images to 32x32, 

[Here](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip)


### Training

```
python3 lisa-classifier.py

pyrhon3 gtsrb-classifier.py
```

### Test

```
demo.ipynb
```

### Reference

- [Traffic Sign Classification using Convolutional Neural Networks](https://github.com/tomaszkacmajor/CarND-Traffic-Sign-Classifier-P2)