# Taiwan_Receipt_App

This is an app that can recognize 8-digit Taiwanese receipts for the receipts lottery.

The first version of this application (first_app) uses a custom made recogizing algorihm, however it wasn't espacially good, so I desided to use a TF CNN instead (version 2), which is much more accurate. OpenCV for webcam management. This source code does not include the final version, but the overall structure of the program.

Requirements:
<br>Python 3.6
<br>OpenCV 4.0
<br>TensorFlow 1.12

_____________________________________________________________________
Project plan:
<br>2019-04-08: Build TF NN to recognize costum dataset

_____________________________________________________________________
Projct log: 
<br>2019-04-01: Created a compiler for data
<br>2019-04-12: Created basic TF NN to recognize dataset
<br>2019-04-13: Did some optimizing.
<br>2019-04-15: Made NN display confidence interval
