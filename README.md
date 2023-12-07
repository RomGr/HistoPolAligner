# Pathology-Guided Quantification of Polarimetric Parameters in Brain Tumors

[![License](https://img.shields.io/pypi/l/PathologyPaper.svg?color=green)](https://github.com/RomGr/PathologyPaper/raw/main/LICENSE)
[![Twitter](https://img.shields.io/twitter/follow/horao_eu?style=flat)](https://twitter.com/horao_eu)
[![CI](https://github.com/RomGr/PathologyPaper/actions/workflows/ci.yml/badge.svg)](https://github.com/RomGr/PathologyPaper/actions/workflows/ci.yml)

by
Romain Gros

The manuscript linked to the present repository is currently under preparation. The present repository contains the code used to generate the results presented in the manuscript.

The study aimed to quantify the polarimetric parameters of neoplastic brain tissue. 

## Abstract

Brain cancer is a significant global health challenge with high mortality rates.
Neuro-oncological surgery is the primary treatment, yet it faces challenges due to glioma invasiveness and to the need to preserve neurological functions.
Achieving a radical resection is unfeasible, emphasizing the need for precise boundary delineation to prevent neurological damage and improve prognosis.
Mueller polarimetry emerges as a promising modality for tumor delineation, demonstrating effectiveness in various tissues.
Our study continues the development of Mueller polarimetry for brain tumor tissue by characterizing its polarimetric properties.
We examined 51 measurements obtained from 29 fresh brain tumor samples collected in 16 differents patients, including diverse tumor types with a strong focus on gliomas.
Using our Imaging Mueller Polarimetry system, we introduced a novel pathology protocol to correlate pathology data with polarimetric measurements, serving as a reliable ground truth for tissue identification.
A custom pipeline facilitated histological and polarimetric image alignment, enabling pathology mask overlay on polarimetry images.
Our analysis quantified depolarization, linear retardance, and optical axis azimuth in fresh neoplastic and healthy brain tissue, emphasizing differentiation between grey and white matter and between tumor-free and neoplastic regions.
We observed significant variations in depolarization for grey and white matter regions, while differences in the linear retardance were observed only within white matter regions of brain tumor tissue.
Notably, we identified pronounced optical axis azimuth randomization within tumor regions, promising advances in brain tumor classification via machine learning.
This study lays the foundation for machine learning-based tumor classification algorithms using polarimetric data, ultimately enhancing brain tumor diagnosis and treatment.


## Software implementation

This GitHub folder documents all the source code used to generate the results and figures in the paper.

The [Jupyter notebooks](http://jupyter.org/) used to generate the results are located in the `notebooks` folder. The data used in this study should be copied from [this link](to be changed), and placed in the `data` folder. Results generated by the code are saved in `results`. Source code used to process the data are loacted in the `src` folder.


## Getting the code

You will in a first time need to create a environnement for imageJ computing in python, following the instructions [here](https://py.imagej.net/en/latest/Install.html).

You can now download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/RomGr/PathologyPaper.git

open a terminal and go to the location of the downloaded folder:

    pip install -e .

After installing the package, you will need to download [Fiji](https://imagej.net/software/fiji/downloads), and copy-paste the files in the `Fiji.app` folder located at the base of the repository.

You will also neeed to install `matlab.engine` package ([instructions here](https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)).


## Align Polarimetry and Histology
The juypter notebooks 1. `AlignHistologyPolarimetrySingleMeasurement.ipynb` and 2. `AlignHistologyPolarimetry.ipynb` are the notebooks used to align:
1. The polarimetry images with the histology images for one sample
2. The polarimetry images with the histology images for all the samples


## Azimuth standard deviation
The juypter notebook `CreateAzimuthStd.ipynb` allows to visualize the local level of noise of the azimuth of the optical axis. The output for this notebook can be found in the different measurements folders, in the `histology` subfolder.

## Get samples information
The juypter notebook `GetSamplesInfo.ipynb` allows to get the samples information from the excel file `samples_paper.xlsx`.

## Verify Alignment Procedure
The juypter notebook `VerifyAlignmentProcedure.ipynb` is a notebook to verify the alignment procedure and to quantify the error between the histology and polarimetry images. The output for this notebook can be found in the different measurements folders, in the `histology` subfolder.

## Quantifiy Parameters Healthy Vs Tumor
The last jupyter notebook `QuantifiyParametersHealthyVsTumor.ipynb` allows to quantify the polarimetric parameters in healthy and tumor regions. The output for this notebook can be found in `results` folder.

## Data
Five subfolders can be found in the `data` folder:
1. `HealthyHuman`: the measurements of healthy human section
2. `HistologyResults`: the histology images of the samples
3. `TumorMeasurements`: the measurements of neoplastic human section

## License
All source code is made available under a BSD license. See `LICENSE` for the full license text.
Special thanks to Stefano Moriconi for the development of the pipeline for processing the Mueller polarimetry images.