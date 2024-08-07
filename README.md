# Train a Deep Learning Model for Land Classification

This project involves training a deep learning model to perform land classification using the urban atlas dataset, featuring architectures like resnet101 and resnet50 as detailed in the corresponding research.

# Dataset:
1. Urban Atlas: [Urban Atlas](https://www.eea.europa.eu/en/datahub/datahubitem-view/e006507d-15c8-49e6-959c-53b61facd873)
2. Download satellite images: [Google Maps static API](https://developers.google.com/maps/documentation/maps-static/overview)

# Dataset processing:
1. Download the dataset of a particular region of a city. Unzip the folder and try to locate the .gpkg file in the subdirectory.
2. Open QGIS and load this .gpkg file in QGIS. Open the layer for this .gpkg file and check (optional) if the attributes table is present for it. 
3. Now export this .gpkg file as a ESRI Shapefile and save the file as countrycode-city (Eg: Berlin City: de-berlin).
4. Use this shapefile to generate a csv file containing latitudes, longitudes, classification etc using Pipeline_to_create_datasets.ipynb.

# Downloading the satellite images:
1. Use the sample_locations and additional_location csv files to download the satellite images using Pipeline_to_acquire_satellite_imagery_from_google_maps.ipynb.
2. Create a Google Maps Static API key using Google Cloud Console. Create a URL signing secret if you will be making more than 25,000 Static Maps API calls for uninterrupted downloads.
3. Download the images and store them in a directory based on their classification from csv file.
4.  
