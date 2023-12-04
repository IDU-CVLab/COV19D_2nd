[![DOI:10.26555/ijain.V9i3.1432.2023.2219765](http://img.shields.io/badge/DOI-10.26555/21681163.2023.2219765-B31B1B.svg)](https://doi.org/10.26555/ijain.v9i3.1432)  

# Database
* The extended version of the COV19-CT-DB database was used. The project includes working on a chest CT-scan series of images to develop an automated algorithmic solution for COVID-19 detection, via [ECCV 2022 Workshop: AI-enabled Medical Image Analysis – Digital Pathology & Radiology/COVID19 - 2nd COV19D Competition](https://mlearn.lincoln.ac.uk/eccv-2022-ai-mia/). <br/>
* The team (IDU-CVLab) is on the leaderboard [here](https://cpb-eu-w2.wpmucdn.com/blogs.lincoln.ac.uk/dist/c/6133/files/2022/07/mia_eccv_2022_leaderboard.pdf).

# Dependencies
- Numpy == 1.19.5
- CV2 == 4.5.4
- Tensorflow == 2.5.0

# Train the Model
* The method proposed is a transfer learning model (Xception model) with a modified output for taking the diagnosis decision of the slices. Hyparperameters tuning was also used. For full details, please check the attached paper.

* To replicate the code properly, you would need a training set of images and a validation set of images.
* The images must be put in the appropriate directories. With that, the directory of training and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
* Please note: this is a binary classification task. To replicate the method on multiple classes, you need to modify the model's output layer to suit your task. 

# Cite
● If you find the method useful, consider citing: <br/>
@article{morani2022covid, <br/>
  title={Covid-19 Detection Using transfer Learning Approach from Computed Temography Images}, <br/>
  author={Morani, Kenan and Balikci, Muhammet Fatih and Altuntas, Tayfun Yigit and Unay, Devrim}, <br/>
  journal={arXiv preprint arXiv:2207.00259}, <br/>
  year={2022} <br/>
}

# Collaboration
* Please get in touch if you wish to collaborate or wish to request the pre-trained models.
