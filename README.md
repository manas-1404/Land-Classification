# Train a Deep Learning Model for Land Classification

This project involves training a deep learning model to perform land classification using the Urban Atlas dataset. The model utilizes architectures such as ResNet101 and ResNet50, based on methodologies from an external research paper.

## Relevant Links
- **Research Paper:** [Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale](https://arxiv.org/abs/1704.02965)
- **GitHub Repository:** [urban-environments](https://github.com/adrianalbert/urban-environments)
- **Research Paper Access:** [urban-environments](https://github.com/adrianalbert/urban-environments)

## Dataset

1. **Urban Atlas**: [Urban Atlas Data](https://www.eea.europa.eu/en/datahub/datahubitem-view/e006507d-15c8-49e6-959c-53b61facd873)
2. **Download Satellite Images**: [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/overview)

## Dataset Processing

1. Download and unzip the dataset for a specific city region, locating the `.gpkg` file.
2. Use QGIS to load and optionally check the `.gpkg` file's attributes table.
3. Export the file as an ESRI Shapefile (e.g., `de-berlin` for Berlin City).
4. Generate a CSV with latitudes, longitudes, and classifications using `Pipeline_to_create_datasets.ipynb`.

## Downloading Satellite Images

![Satellite images of different classes](<images/urban-atlas-images.png>)

1. Download satellite images using `Pipeline_to_acquire_satellite_imagery_from_google_maps.ipynb` with data from `sample_locations` and `additional_location` CSV files.
2. Obtain a Google Maps Static API key and create a URL signing secret for large volume downloads.
3. Store the images in directories based on their classification and remove non-essential classes.

## Approach

### Preprocessing Steps

1. Split the images into training, validation, and testing datasets using `split_data.py`.
2. Preprocess the images using methods in `preprocess.py`.

### Training the Model

1. Select and load the model using `pretrained-resnet50.py`, `pretrained-resnet101.py`, `vgg16.py`, or `resnet.py`.
2. Begin training using `train_model.py`.
3. Use `script1.py`, `script2.py`, and `script3.py` to adjust model layers during training, and run `script.bat` overnight on the terminal.

### Testing the Model

1. Test model accuracy and loss on unseen datasets with `test.py`.
2. Generate a confusion matrix to evaluate performance per class with `confusion_matrix.py`.

### Predictions Using the Model

1. For single image predictions, refer to [Classification-Webapp](https://github.com/manas-1404/Classification-Webapp).
   ![Classification-Webapp](<images/classification-image.jpeg>)
2. For predicting classifications over larger areas like cities, use `Pipeline_model_predictions.ipynb`.
   ![Spatial Distribution of Kaiserslautern](<images/Spatial%20Distribution%20of%20Predicted%20Classes%20for%20Kaiserslautern.png>)
