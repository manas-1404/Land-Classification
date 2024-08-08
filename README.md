# Train a Deep Learning Model for Land Classification
The project involves training a deep learning model to perform land classification using the urban atlas dataset, featuring architectures like resnet101 and resnet50 as detailed in the corresponding research. The project has been developed using an external research paper as reference and some of it's methods. You can find the complete research paper at [urban-environments](https://github.com/adrianalbert/urban-environments).

Research Paper: [Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale](https://arxiv.org/abs/1704.02965)
GitHub: [urban-environments](https://github.com/adrianalbert/urban-environments)

# Dataset:
1. Urban Atlas: [Urban Atlas](https://www.eea.europa.eu/en/datahub/datahubitem-view/e006507d-15c8-49e6-959c-53b61facd873)
2. Download satellite images: [Google Maps static API](https://developers.google.com/maps/documentation/maps-static/overview)

# Dataset processing:
1. Download the dataset of a particular region of a city. Unzip the folder and try to locate the .gpkg file in the subdirectory.
2. Open QGIS and load this .gpkg file in QGIS. Open the layer for this .gpkg file and check (optional) if the attributes table is present for it. 
3. Now export this .gpkg file as a ESRI Shapefile and save the file as countrycode-city (Eg: Berlin City: de-berlin).
4. Use this shapefile to generate a csv file containing latitudes, longitudes, classification etc using Pipeline_to_create_datasets.ipynb.

# Downloading the satellite images:
![Satellite images of different classes](<images/urban-atlas-images.png>)
1. Use the sample_locations and additional_location csv files to download the satellite images using Pipeline_to_acquire_satellite_imagery_from_google_maps.ipynb.
2. Create a Google Maps Static API key using Google Cloud Console. Create a URL signing secret if you will be making more than 25,000 Static Maps API calls for uninterrupted downloads.
3. Download the images and store them in a directory based on their classification from csv file.
4. Initially you will find a lot of different classes in the directory. Delete the classes which aren't essential for your research.

# Approach
# Preprocessing Steps
1. Split the satellite images into training, valuation, and testing datasets using the split_data.py.
2. Use the preprocess.py methods for preprocessing the images

# Training the model
1. Choose the model which you want to use for training. You can load the pretrained resnet50 and resnet101 model using pretrained-resnet50.py and pretrained-resnet101.py respectively. You can also use the vgg16.py, resnet.py to build a vgg16, resnet model for training. 
2. Load the model in train_model.py and start the training of the model. 
3. You can use script1.py, script2.py, script3.py to unfreeze different layers of the model while training. You can use script.bat on the terminal to run them overnight.

# Testing the model
1. You can test the accuracy & loss of the model using an unseen dataset. Simply load the model and run test.py.
2. You can see the performance of the model in classifying each class by generating a confusion matrix. Simply load the model and run confusion_matrix.py.

# Predictions using model
You can use the model for making predictions of the images. 
1. If you simply want to predict the classification type of a single image, then you can refer to my [Classification-Webapp](https://github.com/manas-1404/Classification-Webapp) for making predictions.
    ![Classification-Webapp](<images/classification-image.jpeg>)
2. If you wish to make a prediction on a larger area like a city, then you can easily execute the Pipeline_model_predictions.ipynb for generating a spatial distribution of the city.
    ![Spatial Distribution of Kaiserslautern](<images/Spatial%20Distribution%20of%20Predicted%20Classes%20for%20Kaiserslautern.png>)