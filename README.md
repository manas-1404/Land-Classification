# Train a Deep Learning Model for Land Classification

This project involves training a deep learning model to perform land classification using the Urban Atlas dataset. The model utilizes architectures such as ResNet101 and ResNet50, based on methodologies from an external research paper.

## References
- **Research Paper:** [Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale](https://arxiv.org/abs/1704.02965)
- **GitHub Repository:** [urban-environments](https://github.com/adrianalbert/urban-environments)

## Dataset Sources

1. **Urban Atlas**: [Urban Atlas Data](https://www.eea.europa.eu/en/datahub/datahubitem-view/e006507d-15c8-49e6-959c-53b61facd873)
2. **Download Satellite Images**: [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/overview)

## Custom Packages Requirements
The project requires you to use certain custom-made packages for downloading the datasets and processing them. These packages are 7 years old with all of the code present in the deprecated Python2 version and there's no community to support & develop the packages. I updated the packages to newer Python3 version which is compatible with the current project.  
1. Pysatapi: [pysatapi](https://github.com/manas-1404/pysatapi) - Advisable to install the package directly through the master branch and appending it to the Python path in case of 'ModuleNotFoundError'. (Implementation provided in `Pipeline_to_acquire_satellite_imagery_from_google_maps.ipynb`)
2. Pysatml: [pysatml](https://github.com/manas-1404/pysatml) - Advisable to install the package directly through the master branch and appending it to the Python path in case of 'ModuleNotFoundError'. In case, you face gdal 'ModuleNotFoundError' error then make sure to import it like this 'from osgeo import gdal, osr, gdalconst'.

## Virtual environment set-up
The project can be easily replicated with the help of a conda virtual environment. Based on my system configuration, I have decided to use Python 3.9 version. 

1. Install miniconda easily from the official [Miniconda installation guide](https://docs.anaconda.com/miniconda/)
2. Open the miniconda prompt. 
3. Navigate to the project directory /Land-Classification
3. Set-up the virtual environment using "conda create --name landClassification python=3.9"
4. Activate the virtual environment using "conda activate landClassification"
5. You can deactivate it later with the following command "conda deactivate"

## Virtual environment set-up
The project can be easily replicated with the help of a conda virtual environment. Based on my system configuration, I have decided to use Python 3.9 version. 

1. Install miniconda easily from the official [Miniconda installation guide](https://docs.anaconda.com/miniconda/)
2. Open the miniconda prompt. 
3. Navigate to the project directory `/Land-Classification`
4. Set-up the virtual environment using:

   ```bash
conda create --name landClassification python=3.9
<button onclick="copyToClipboard('#code1')">Copy Command</button>
<pre><code id="code1">conda create --name landClassification python=3.9</code></pre>
```

## Tensorflow requirements for set-up
Installing tensorflow in older versions of windows (Windows native: Windows 7 or higher (64-bit)) was the biggest hassle. The easiest way to install tensorflow with GPU support is by utilizing Miniconda. It is also the easiest way to install the required software (cuDNN and CUDA) especially for the GPU setup.

It is very important to ensure that your tensorflow version is always compatible with the python version, cuDNN, CUDA versions. You can check the complete [compatibility list](https://www.tensorflow.org/install/source#gpu)

1. Activate the conda virtual environment by using "conda activate <name of virtual environment>
2. Install [NVIDIA GPU](https://www.nvidia.com/Download/index.aspx) driver based on your system's GPU specifications.
3. Install CUDA and cuDNN with the help of conda "conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0". If you are replicating this project, then you can directly install the CUDA and cuDNN versions mentioned earlier.
4. Install the tensorflow version which is COMPATIBLE with CUDA, cuDNN, and your Python version. If you are replicating this project, then you can directly install tensorflow 2.10 version with "pip install tensorflow==2.10".

If you are not installing the above mentioned versions of tensorflow, then I'd highly recommend you check out the official [tensorflow installation guide](https://www.tensorflow.org/install/pip#windows-wsl2_1)

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


### 2. HTML and JavaScript Section
```html
<script>
function copyToClipboard(element) {
  var copyText = document.querySelector(element).innerText;
  navigator.clipboard.writeText(copyText).then(() => {
    alert('Copied to clipboard');
  });
}
</script>
```