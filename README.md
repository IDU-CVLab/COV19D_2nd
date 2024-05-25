[![DOI:10.1155/2024/9962839](http://img.shields.io/badge/DOI-10.1155/2024/9962839.svg)](https://doi.org/10.1155/2024/9962839)
AND  
[![DOI:10.26555/ijain.V9i3.1432.2023.2219765](http://img.shields.io/badge/DOI-10.26555/21681163.2023.2219765-B31B1B.svg)](https://doi.org/10.26555/ijain.v9i3.1432)  
# Database
* The extended version of the COV19-CT-DB database was used. The project includes working on a chest CT-scan series of images to develop an automated algorithmic solution for COVID-19 detection, via [ECCV 2022 Workshop: AI-enabled Medical Image Analysis – Digital Pathology & Radiology/COVID19 - 2nd COV19D Competition](https://mlearn.lincoln.ac.uk/eccv-2022-ai-mia/). The team (IDU-CVLab) is on the leaderboard [here](https://cpb-eu-w2.wpmucdn.com/blogs.lincoln.ac.uk/dist/c/6133/files/2022/07/mia_eccv_2022_leaderboard.pdf).  
* The full methodology with image processing techniques was retrained and tested on the extended version of the COV19-CT Database, which was distributed for the [forth run of the competetion](https://mlearn.lincoln.ac.uk/ai-mia-cov19d-competition/).

# Dependencies
- Numpy == 1.19.5
- CV2 == 4.5.4
- Tensorflow == 2.5.0

# Methodology
This method can be applied in two different ways; With images processing and without (using the two different codes in the repository):  
**With images processing (Optional).** Uppermost and lower most slcies of each CT scans were removed with 60% of the slices in each CT scan is kept. Next, Slices of the CT scan were manually cropped to better localize the Region of Interste (ROI), which is the two lung areas.  
**Trnasfer Learning-Based Classifier.** To take the diagnostic desicions at slices level, a transfer learning model (Xception model) with a modified output was deployed.    
* To replicate the code properly, you would need a training set of images and a validation set of images.
* The images must be put in the appropriate directories. With that, the directory of training and validation sets in the codes should be changed to match your directories. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
* Please note: this is a binary classification task. To replicate the method on multiple classes, you need to modify the model's output layer to suit your task.
* For full details, please check the attached papers. 
* Kindly inform the organization owner if you wish to obtain the pretrained models in this study.
  
# Cite
If you find these methods useful, kindly consider citing: <br/>
> @article{IJAIN1432|to_array:0,  
	author = {Kenan Morani and Esra Kaya Ayana and Devrim Unay},  
	title = {Covid-19 detection using modified xception transfer  learning approach from computed tomography images},  
	journal = {International Journal of Advances in Intelligent Informatics},
	volume = {9},  
	number = {3},  
	year = {2023},  
	issn = {2548-3161},	pages = {524--536},	doi = {10.26555/ijain.v9i3.1432},  
	url = {http://ijain.org/index.php/IJAIN/article/view/1432}  
}

> @misc{morani2023covid19,  
      title={COVID-19 Detection Using Slices Processing Techniques and a Modified Xception Classifier from Computed Tomography Images},   
      author={Kenan Morani},  
      year={2023},  
      eprint={2312.07580},  
      archivePrefix={arXiv},  
      primaryClass={eess.IV}  
}
