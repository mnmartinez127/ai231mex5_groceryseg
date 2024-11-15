# ai231mex5_groceryseg
AI 231 Machine Exercise 5 - Grocery Items Detection / Segmentation

Code for Machine Exercise 5: an application to perform detection and segmentation on images or a webcam stream

This program uses YOLOv11 trained on a dataset of 24 classes of grocery items.

## Model Training

## Model Inference
### Instructions
This application requires Python and streamlit to be installed. Install streamlit using:
```
pip install streamlit
```

Clone the repository and run the following command:
```
streamlit run mex5-infer.py
```
Then open a web browser and go to:
```

```
### Using the app
By default, the application is set to detect and segment all instances of the 24 known grocery items that can be seen via your device's webcam.
The raw webcam input can be seen in the left window, while the results of the model inference (bounding boxes, labels, and segmentation masks) are found in the right window.



### Training on custom data


### Training details
The model used in this application is a YOLO11-m model fine-tuned on a dataset of 24 classes of grocery items, compiled and annotated by the AI 231 2024-2025 1st Semester class of the University of the Philippines Diliman.
Training was performed over 200 epochs with a 
The patch sub-directories, **12_nestle_all_purpose_cream** and **ketchup**, are needed to complete the dataset, however the training script will still run if either of these are missing. However, the **dataset** folder is required.
The dataset layout, with patches, is as follows:

data/detection/
└───grocery
    ├───12_nestle_all_purpose_cream
    │   ├───images
    │   │   ├───train
    |   |   |   ├───120001.jpg
    |   |   |   └───...
    │   │   └───val
    |   |       ├───120005.jpg
    |   |       └───...
    │   └───labels
    |           ├───train_nestle.json
    |           └───val_nestle.json
    ├───dataset
    │   ├───annotations
    |   |       ├───instances_train.json
    |   |       └───instances_val.json
    │   └───images
    │       ├───train
    |       |   ├───010001.jpg
    |       |   └───...
    │       └───val
    |           ├───010012.jpg
    |           └───...
    └───ketchup
        ├───images
        |   ├───train
        |   |   ├───100002.jpg
        |   |   └───...
        |   └───val
        |       ├───100001.jpg
        |       └───...
        └───labels
                ├───train_ketchup_annotations.json
                └───val_ketchup_annotations.json

The 
### Evaluation results

The training script runs evaluation alongside model generation. The model used in the application has the following metrics:


