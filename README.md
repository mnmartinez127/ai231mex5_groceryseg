# ai231mex5_groceryseg
AI 231 Machine Exercise 5 - Grocery Items Detection / Segmentation

Code for Machine Exercise 5: an application to perform detection and segmentation on images or a webcam stream

This program uses YOLOv11 trained on a dataset of 24 classes of grocery items.

## Model Training

## Model Inference
### Instructions
```
pip install streamlit
```

Clone the repository and navigate to the repository folder in a terminal. This application requires Python and several dependencies to be installed. Install them using:
```
pip install -r requirements.txt
```

To open the inference application, run the following command:
```
streamlit run infer.py -- -a [ADDRESS] -p [PORT]
```
Both `-a [ADDRESS]` and `-p [PORT]` are optional and default to the UP Diliman AI Program server and port 8002, respectively. To change these, use the `-a` and `-p` options as shown above. Ex: `streamlit run infer.py -- -a 231.120.24.230 -p 8086`

Once the application is active, you can open a web browser and go to:
```
http://localhost:8006
```
to open the application user interface.

The application can be used on its own. However, it can also be used as a client to the included server script. To run the server, open a terminal inside the repository folder, then run:
```
python server.py -p [PORT]
```
This will run the server component on a specified port Ex: `python server.py -p 8000` will run the server on port 8000 of your device. `-p [PORT]` is optional and will default to port 8002 being used.
If you are using ssh on a server, you can use:
```
nohup sh -c 'python server.py' > server_out.txt 2>&1 &
```
to keep your server alive even if you disconnect.
### Using the app
The models included in this application are set to detect and segment all instances of 24 known grocery items that can be seen via your device's webcam. Click on the Start button  on the right side of the page to open your webcam (make sure to enable camera permissions!), and the application will show your video stream with all images annotated.

Common options are found on the left side of the page. More advanced options, including the option to perform local inference, can be found by opening the left sidebar. Note that the application is set to upload the webcam stream to the server for inference, so if you want to do inference locally (or you don't want to send your stream to a server), you have to set it in the advanced options.

### Training details
The model used in this application is a YOLO11 segmentation model fine-tuned on a dataset of 24 classes of grocery items, compiled and annotated by the AI 231 2024-2025 1st Semester class of the University of the Philippines Diliman.
Training was performed over 500 epochs using the **yolo11m-seg** model from Ultralytics as a base. Two more models were also trained under the same settings using the **yolo11s-seg** and **yolo11n-seg** models as a base, presented in decreasing order of size.
The patch sub-directories, **12_nestle_all_purpose_cream** and **ketchup**, are needed to complete the dataset, however the training script will still run if either of these are missing. However, the **dataset** folder is required.
The dataset layout, with patches, is as follows:
```
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
```

### Training on dataset
Make sure that your dataset is in the correct format as listed above.
To run the training script, open a terminal inside the repository folder and run:
```
python trainer.py
```
If you are using ssh on a server, you can use:
```
tmux
nohup sh -c 'python trainer.py' > training_results.txt 2>&1 &
tmux detach
```
to keep your server alive even if you disconnect. The use of tmux is required to keep the model alive.
The resulting models are stored in the **models** folder for use by the inference application.

### Evaluation results
The training script runs evaluation alongside model generation. The models used in the application have the following metrics:


| Model     | Precision | Recall  | mAP50   | mAP50-95 |
|-----------|-----------|---------|---------|----------|
| YOLO 11 N | 0.95788   | 0.92526 | 0.93911 | 0.87137  |
| YOLO 11 S | 0.95634   | 0.9161  | 0.93181 | 0.84751  |
| YOLO 11 M | 0.95386   | 0.92581 | 0.9384  | 0.8758   |

Note that precision is higher than recall, thus the model is better at distinguishing between different classes than recognizing an object as belonging to a certain class.