#import statements
import os
import sys
import shutil
import json
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO, SAM
    from ultralytics.data.converter import convert_coco
except ImportError:
    os.system("pip install opencv-python opencv-python-contrib")
    os.system("pip install 'numpy<2.0'")
    os.system("pip install ultralytics")
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

#Training data folders
data_dirs = []
train_dirs = []
val_dirs = []

annotation_dirs = []
train_annotation_dirs = []
val_annotation_dirs = []

root_dir = os.path.join(os.getcwd())#,"..")
data_root_dir = os.path.join("data","detection","grocery")
data_root_dir = os.path.join(root_dir,data_root_dir)
#Master dataset
data_dirs.append(data_root_dir,"dataset") #path to master dataset
train_dirs.append(os.path.join(data_dirs[0],"images","train"))
val_dirs.append(os.path.join(data_dirs[0],"images","val"))
annotation_dirs.append(os.path.join(data_dirs[0],"annotations"))
train_annotation_dirs.append(os.path.join(annotation_dirs[0],"instances_train.json"))
val_annotation_dirs.append(os.path.join(annotation_dirs[0],"instances_val.json"))

def add_patch(class_path,
            patch_train_path = os.path.join("images","train"),
            patch_val_path = os.path.join("images","val"),
            patch_annotations_path = "labels",
            patch_train_annotations_name = "instances_train.json",
            patch_val_annotations_name = "instances_val.json"):
    if os.path.exists(class_path):
        data_dirs.append(class_path)
        train_dirs.append(os.path.join(data_dirs[-1],patch_train_path))
        val_dirs.append(os.path.join(data_dirs[-1],patch_val_path))
        annotation_dirs.append(os.path.join(data_dirs[-1],patch_annotations_path))
        train_annotation_dirs.append(os.path.join(annotation_dirs[-1],patch_train_annotations_name))
        val_annotation_dirs.append(os.path.join(annotation_dirs[-1],patch_val_annotations_name))

#add patches here

#Class 10 - Ketchup
class10_path = os.path.join("data_root_dir","..","..","ketchup")
add_patch(class10_path,patch_annotations_path = "labels",
          patch_train_annotations_name="train_ketchup_annotations.json",
          patch_val_annotations_name="val_ketchup_annotations.json")

#Class 12 - Nestle All Purpose Cream
class12_path = os.path.join("data_root_dir","..","..","ketchup")
add_patch(class12_path,patch_annotations_path = "labels",
          patch_train_annotations_name="train_nestle.json",
          patch_val_annotations_name="val_nestle.json")

#Complete dataset directory
new_data_dir = os.path.join(root_dir,"dataset")
new_train_dir = os.path.join(new_data_dir,"images","train")
new_val_dir = os.path.join(new_data_dir,"images","val")

new_annotation_dir = os.path.join(new_data_dir,"annotations")
new_train_annotation_dir = os.path.join(new_annotation_dir,"instances_train.json")
new_val_annotation_dir = os.path.join(new_annotation_dir,"instances_val.json")

label_dir = os.path.join(new_data_dir,"labels")
model_dir = os.path.join(root_dir,"models")
yaml_dir = os.path.join(root_dir,"grocery.yaml")




#Skip dataset preprocessing if YAML file is present
if not os.path.exists(yaml_dir):
    #Combine annotations to a single directory
    def get_annotations(annotation_dirs,new_annotation_dir):
        new_annotations = {"images":[],"annotations":[],"categories":[]}
        ids = {"images":set(),"annotations":set(),"categories":set()}
        
        for annotation_dir in annotation_dirs:
            with open(annotation_dir,"r",encoding="utf-8") as f:
                annotations = json.load(f)
                for column in ["images","annotations","categories"]:
                    for i in range(len(annotations[column])):
                        if annotations[column][i]["id"] not in ids[column]:
                            ids[column].add(annotations[column][i]["id"])
                            new_annotations[column].append(annotations[column][i])

        with open(new_annotation_dir,"w",encoding="utf-8") as f:
            json.dump(new_annotations,f)

    #make a local copy of the annotations
    if not os.path.isdir(new_train_annotation_dir):
        os.makedirs(new_annotation_dir,exist_ok=True)
        get_annotations(train_annotation_dirs,new_train_annotation_dir)
    if not os.path.isdir(new_val_annotation_dir):
        os.makedirs(new_annotation_dir,exist_ok=True)
        get_annotations(val_annotation_dirs,new_val_annotation_dir)
    print("Compiled annotations!")

    #Combine images to a single directory
    def get_images(data_dirs,new_data_dir):
        for data_dir in data_dirs:
            for image in os.listdir(data_dir):
                shutil.copy2(os.path.join(data_dir,image),os.path.join(new_data_dir,image))

    # make a local copy of the dataset
    if not os.path.isdir(new_train_dir):
        os.makedirs(new_train_dir,exist_ok=True)
        get_images(train_dirs,new_train_dir)
    if not os.path.isdir(new_val_dir):
        os.makedirs(new_val_dir,exist_ok=True)
        get_images(val_dirs,new_val_dir)
    print("Compiled dataset images!")

    #fix annotations after merging
    def fix_annotations(annotation_path,fixed_annotation_path,debug=False):

        with open(annotation_path,"r",encoding="utf-8") as f:
            annotations = json.load(f)
        image_dims = {}
        for i in range(len(annotations["images"])):
            width = annotations["images"][i]["width"]
            height = annotations["images"][i]["height"]
            image_dims[annotations["images"][i]["id"]] = (width,height)

        for i in range(len(annotations["annotations"])):
            ann_dim = image_dims[annotations["annotations"][i]["image_id"]]

            if "segmentation" in annotations["annotations"][i].keys():
                #Clip segmentation polygon
                clipped_anns = []
                for j in range(len(annotations["annotations"][i]["segmentation"])):
                    clipped_pts = []
                    for k in range(len(annotations["annotations"][i]["segmentation"][j])):
                        clipped_pts.append(min(max(0,annotations["annotations"][i]["segmentation"][j][k]),ann_dim[k%2]))
                    clipped_anns.append(clipped_pts)
                annotations["annotations"][i]["segmentation"] = clipped_anns
            if "bbox" in annotations["annotations"][i].keys():
                #Clip bounding box
                clipped_pts = []
                for j in range(len(annotations["annotations"][i]["bbox"])):
                    clipped_pts.append(min(max(0,annotations["annotations"][i]["bbox"][j]),ann_dim[j%2]))
                annotations["annotations"][i]["bbox"] = clipped_pts

        with open(fixed_annotation_path,"w",encoding="utf-8") as f:
            json.dump(annotations,f)
        if debug:
            print("Clipping complete!")

    fix_annotations(new_train_annotation_dir,new_train_annotation_dir)
    fix_annotations(new_val_annotation_dir,new_val_annotation_dir)
    print("Fixed annotations!")

    #Convert annotations to YOLO format
    if not os.path.exists(os.path.join(new_data_dir,"yolo_annotations")):
        convert_coco(new_annotation_dir,os.path.join(new_data_dir,"yolo_annotations"),use_segments=True,cls91to80=False)
    os.makedirs(os.path.join(new_data_dir,"labels","train"),exist_ok=True)
    for ann in os.listdir(os.path.join(new_data_dir,"yolo_annotations","labels","train")):
        shutil.copy2(os.path.join(new_data_dir,"yolo_annotations","labels","train",ann),os.path.join(new_data_dir,"labels","train",ann))
    os.makedirs(os.path.join(new_data_dir,"labels","val"),exist_ok=True)
    for ann in os.listdir(os.path.join(new_data_dir,"yolo_annotations","labels","val")):
        shutil.copy2(os.path.join(new_data_dir,"yolo_annotations","labels","val",ann),os.path.join(new_data_dir,"labels","val",ann))
    shutil.rmtree(os.path.join(new_data_dir,"yolo_annotations"))
    print("Converted COCO dataset to YOLO format.")

    #fix image orientation after merging annotations and merging images
    def fix_image_orientation(data_dir,annotation_path,clockwise=False,debug=False):
        with open(annotation_path,"r",encoding="utf-8") as f:
            annotations = json.load(f)
        image_dims = {}
        for i in range(len(annotations["images"])):
            width = annotations["images"][i]["width"]
            height = annotations["images"][i]["height"]
            image_dims[annotations["images"][i]["id"]] = (width,height)
            image_path = os.path.join(data_dir,annotations["images"][i]["path"])
            image = cv2.imread(image_path)
            if width != image.shape[1] or height != image.shape[0]:
                if debug:
                    print(f"Mismatch on {image_path}")
                    print(f"w: {width}={image.shape[1]} | h: {height}={image.shape[0]}")
                if width == image.shape[0] and height == image.shape[1]:
                    if clockwise:
                        image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
                    else: #rotate -90 degrees
                        image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
                    if debug:
                        print(f"Correcting via {'clockwise' if clockwise else 'counterclockwise'} rotation")
                        print(f"w: {width}={image.shape[1]} | h: {height}={image.shape[0]}")
                    cv2.imwrite(image_path,image)
                else:
                    image = cv2.resize(image,(width,height))
                    if debug:
                        print(f"Correcting via rescaling")
                        print(f"w: {width}={image.shape[1]} | h: {height}={image.shape[0]}")
                    cv2.imwrite(image_path,image)
                    
    fix_image_orientation(new_data_dir,new_train_annotation_dir)
    fix_image_orientation(new_data_dir,new_val_annotation_dir)
    print("Fixed dataset image orientations.")

    #get categories after merging annotations
    def get_categories(annotation_dir,custom_categories = None):
        
        with open(annotation_dir,"r",encoding="utf-8") as f:
            annotations = json.load(f)
        categories = {}
        #check if the categories start at index 0 and adjust annotations if not
        offset_categories = 1 if all([int(annotations["categories"][i]["id"]) for i in range(len(annotations["categories"]))]) else 0
        for i in range(len(annotations["categories"])):
            categories[int(annotations["categories"][i]["id"])-offset_categories] = annotations["categories"][i]["name"]
        if isinstance(custom_categories,dict) and custom_categories:
            for k,v in custom_categories.items():
                categories[k] = v
        categories = [categories.get(i,f"CLASS_{i}_") for i in range(max(categories.keys())+offset_categories)]
        print(categories)
        return categories

    custom_categories = {
    0: "Coke Zero",
    1: "Eden cheese",# (box,sachet)",
    2: "Kitkat",
    3: "Nescafe 3-in-1 original",# (single, twin pack)",
    4: "Alaska Classic",# (377g can)",
    5: "Simply Pure Canola Oil",
    6: "Purefoods Corned Beef",
    7: "Garlic",
    8: "Lucky Me Pancit Canton",
    9: "UFC Banana Ketchup",
    10: "Lemon",
    11: "Nestle All-Purpose Cream",# (250 ml)",
    12: "Lady's Choice Real Mayonnaise",# (220 ml jar)",
    13: "Peanut Butter",
    14: "Pasta",# (spaghetti, macaroni)",
    15: "del monte green pineapple juice",# (fiber, ace)",
    16: "Rebisco + Skyflakes Crackers",# (singles, transparent packaging)",
    17: "555 Sardines",# (can)",
    18: "Sunsilk Shampoo",# (Pink)",
    19: "Dove Soap",# (relaxing lavander)",
    20: "Silver Swan Soy Sauce",# (385 mL transluscent)",
    21: "Colgate Toothpaste",# (Advanced White Value Pack 2 Tubes)",
    22: "Century Tuna",# (canned, short and tall, white color)",
    23: "Green Cross Ethyl Alcohol",
    }

    train_categories = get_categories(new_train_annotation_dir,custom_categories)
    val_categories = get_categories(new_val_annotation_dir,custom_categories)
    print("Obtained categories.")


    #generate YAML file
    #Delete the YAML to regenerate the dataset
    with open(yaml_dir, "w",encoding="utf-8") as f:
        f.write(f"path: ../{os.path.relpath(new_data_dir,root_dir).replace('\\','/')}"+"\n")
        f.write(f"train: {os.path.relpath(new_train_dir,new_data_dir).replace('\\','/')}"+"\n")
        f.write(f"val: {os.path.relpath(new_val_dir,new_data_dir).replace('\\','/')}"+"\n")
        f.write("test:\n")
        f.write("# Classes\n")
        f.write("names:\n")
        for i in range(len(train_categories)):
            f.write(f"  {i}: {train_categories[i]}"+"\n")
    print("Preprocessing complete!")

#Train the model(s) on the dataset
os.makedirs(model_dir,exist_ok=True)

if torch.cuda.is_available():
    devices = list(range(torch.cuda.device_count()))
    #Assign gpu as first argument
    #Do not assign to use all gpus
    if len(sys.argv) > 1 and int(sys.argv[1]) in devices:
        if int(sys.argv[1]) >= 0 and int(sys.argv[1]) < torch.cuda.device_count():
            devices = int(sys.argv[1])
        elif int(sys.argv[1]) == -1:
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = 0
    else:
        devices = 0
else:
    devices = "cpu"

models = {
    "YOLOv11 Segmentation N": "yolo11n-seg.pt",
    "YOLOv11 Segmentation S": "yolo11s-seg.pt",
    "YOLOv11 Segmentation M": "yolo11m-seg.pt",
    #"YOLOv11 Segmentation L": "yolo11l-seg.pt",
}




for k,v in models.items():
    print(f"{k}: {os.path.join(model_dir,v)}")

    try:
        #reload last checkpoint if exists
        if os.path.isfile(os.path.join(model_dir,k,"weights","last.pt")):
            model = YOLO(os.path.join(model_dir,k,"weights","last.pt"),task="segment")
            trained_model = model.train(resume=True)
        else:
            model = YOLO(v,task="segment")
            trained_model = model.train(
            data=yaml_dir,
            epochs=500,
            imgsz=640,
            batch=16,
            patience=20,
            optimizer="auto",
            lr0=1e-3,
            lrf=0.01,
            device=devices,
            seed=42,
            val=True,
            plots=True,
            dropout=0.05,
            save=True,
            save_period=1,
            cache=True,
            project=model_dir,
            name=k,
            exist_ok=True,
            )
    except Exception:
        print(f"{k} failed to train!")
        continue