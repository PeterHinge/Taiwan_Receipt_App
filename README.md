# Taiwan_Receipt_App

This is an app that can recognize 8-digit Taiwanese receipts for the receipts lottery.

The first version of this application (first_app) uses a custom made recogizing algorihm, however it wasn't espacially good, so I desided to use a TF CNN instead (version 2), which is much more accurate. OpenCV for webcam management.

Requirements:

Python 3.6

OpenCV 4.0

TensorFlow 1.12

_____________________________________________________________________
Project plan:
2019-04-08: Build TF NN to recognize costum dataset

_____________________________________________________________________
Projct log: 
2019-04-01: Created a compiler for data
2019-04-12: Created basic TF NN to recognize dataset
2019-04-13: Did some optimizing.
2019-04-15: Made NN display confidence interval
