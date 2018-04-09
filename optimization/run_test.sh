#!/bin/bash

python classify.py --model GoogleNet ../../googlenet.npy test/speedLimit45.jpg

python classify.py --model GoogleNet ../../googlenet.npy output/speedLimit45.png

