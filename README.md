# Introduction
[![DOI:10.48550/arXiv.2111.11191](http://img.shields.io/badge/DOI-10.1101/2021.01.08.425840-B31B1B.svg)](https://arxiv.org/abs/2207.00259)

* The extended version of the COV19-CT-DB database was used. The project includes working on a chest CT-scan series of images to develop an automated algorithmic solution for COVID-19 detection, via [ECCV 2022 Workshop: AI-enabled Medical Image Analysis – Digital Pathology & Radiology/COVID19 - 2nd COV19D Competition](https://mlearn.lincoln.ac.uk/eccv-2022-ai-mia/). <br/>
* The method proposed is a transfer learning model (Xception model) with a modified output for taking the diagnosis decision of the slices. Hyparperameters tuning was also used.
* The team (IDU-CVLab) is on the leaderboard [here](https://cpb-eu-w2.wpmucdn.com/blogs.lincoln.ac.uk/dist/c/6133/files/2022/07/mia_eccv_2022_leaderboard.pdf).

To replicate the code, the following must be noted:
1. To run the code properly you would need a training set of images and a validation set of images.
2. The images must be put in the appropriate directories. With that, the directory of training and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator <br /> 

# Dependencies
- Numpy == 1.19.5
- CV2 == 4.5.4
- Tensorflow == 2.5.0


# Citation
● If you find the method useful, consider citing: <br/>
@article{morani2022covid, <br/>
  title={Covid-19 Detection Using transfer Learning Approach from Computed Temography Images}, <br/>
  author={Morani, Kenan and Balikci, Muhammet Fatih and Altuntas, Tayfun Yigit and Unay, Devrim}, <br/>
  journal={arXiv preprint arXiv:2207.00259}, <br/>
  year={2022} <br/>
}

# Colaboration
* Please get in touch if you wish to collaborate or wish to request the pre-trained models.
