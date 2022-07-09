# COV19D_2nd
[![DOI:10.48550/arXiv.2111.11191](http://img.shields.io/badge/DOI-10.1101/2021.01.08.425840-B31B1B.svg)](https://arxiv.org/abs/2207.00259)

The project includes working on a chest CT-scan series of images to develop an automated algorithm for COVID-19 detection, via [ECCV 2022 Workshop: AI-enabled Medical Image Analysis – Digital Pathology & Radiology/COVID19 - 2nd COV19D Competition](https://mlearn.lincoln.ac.uk/eccv-2022-ai-mia/).

To replicate the codes, the following must be noted:
1. To run the code properly you would need a training set of images and a validation set of images.
2. The images must be put in the appropriate directories. With that, the directory of training and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator <br /> 

<b> Dependencies: </b><br/>
- Numpy == 1.19.5
- CV2 == 4.5.4
- Tensorflow == 2.5.0
