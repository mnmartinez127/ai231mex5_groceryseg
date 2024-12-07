#import statements
import os
import argparse
import shutil
import json
import re
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.data.converter import convert_coco
except ImportError:
    os.system("pip install opencv-python opencv-contrib-python")
    os.system("pip install 'numpy<2.0'")
    os.system("pip install ultralytics")
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.data.converter import convert_coco

DEVICE_LIST = list(range(torch.cuda.device_count()))
NUM_EPOCHS = 500
MAX_RETRIES = 1
MODEL_FOLDER = "models"
TRAINER_VERSION = 6
MODEL_CODES = ["N","S","M","L","X"]
BATCH_SIZES = [256,128,64,32,16,8,4,2,1]
parser = argparse.ArgumentParser(description="Specify the device to use and number of epochs in training.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a",action='store_true',help="Apply augmentations")
parser.add_argument("-b",default=MODEL_CODES,nargs="+",type=str,help="Set the model codes in the order of training. Available codes are N S M L X")
parser.add_argument("-d",default=[0 if torch.cuda.is_available() else -1],nargs="+",type=int,help="Training Device. Set to -1 to ONLY use cpu")
parser.add_argument("-n",default=NUM_EPOCHS,type=int,help="Number of Epochs")
parser.add_argument("-o",default=MODEL_FOLDER,type=str,help="Specify name of models output folder")
parser.add_argument("-l",default="",type=str,help="Path to transfer learning model")
parser.add_argument("-r",action='store_true',help="Forcibly regenerate dataset")
parser.add_argument("-s",default=BATCH_SIZES,nargs="+",type=int,help="Set the batch sizes to attempt training. Supported sizes are 256 128 64 32 16 8 4 2 1")
parser.add_argument("-t",action='store_true',help="Enable Training")
parser.add_argument("-v",default=list(range(TRAINER_VERSION+1)),nargs="+",type=int,help="""
Specify the version/s of the dataset to use for transfer learning.
Leave blank to use all components, input each version number to use it
Example: 0 1 2 3 will use the first four versions of the dataset
0: base dataset
1: additional images for class 10 and 12
2: additional images for all classes
3: additional images for silver swan
4: additional images for nestle
5: additional low-light images for all classes
6: additional images for del monte pineapple
""")
parser.add_argument("-w",default=4,type=int,help="Set the number of dataloader workers, defaults to 4")
args = parser.parse_args()

#Training data folders
data_dirs = []
train_dirs = []
val_dirs = []
annotation_dirs = []
train_annotation_dirs = []
val_annotation_dirs = []

root_dir = os.path.join(os.getcwd())
data_root_dir = os.path.join("data")
data_root_dir = os.path.join(root_dir,data_root_dir)

#Complete dataset directory
new_data_dir = os.path.join(root_dir,"dataset")
new_train_dir = os.path.join(new_data_dir,"images","train")
new_val_dir = os.path.join(new_data_dir,"images","val")

new_annotation_dir = os.path.join(new_data_dir,"annotations")
new_train_annotation_dir = os.path.join(new_annotation_dir,"instances_train.json")
new_val_annotation_dir = os.path.join(new_annotation_dir,"instances_val.json")

label_dir = os.path.join(new_data_dir,"labels")
model_dir = os.path.join(root_dir,args.o)
yaml_dir = os.path.join(root_dir,"grocery.yaml")

def verify_dirs():
    list_dirs = [data_dirs,train_dirs,val_dirs,annotation_dirs,train_annotation_dirs,val_annotation_dirs]
    name_dirs = ["DATA","TRAIN","VAL","ANNOTATION","TRAIN ANNOTATION","VAL ANNOTATION"]
    for i in range(len(name_dirs)):
        print(name_dirs[i])
        for j in range(len(list_dirs[i])):
            print(list_dirs[i][j])
verify_dirs()

if __name__ == "__main__":
    #Skip dataset preprocessing if YAML file is present
    if args.r or not os.path.isfile(yaml_dir):
        def add_patch(class_path,
                    patch_train_path = os.path.join("images","train"),
                    patch_val_path = os.path.join("images","val"),
                    patch_annotations_path = "annotations",
                    patch_train_annotations_name = "instances_train.json",
                    patch_val_annotations_name = "instances_val.json"):
            if os.path.exists(class_path):
                data_dirs.append(class_path)
                train_dirs.append(os.path.join(data_dirs[-1],patch_train_path))
                val_dirs.append(os.path.join(data_dirs[-1],patch_val_path))
                annotation_dirs.append(os.path.join(data_dirs[-1],patch_annotations_path))
                train_annotation_dirs.append(os.path.join(annotation_dirs[-1],patch_train_annotations_name))
                val_annotation_dirs.append(os.path.join(annotation_dirs[-1],patch_val_annotations_name))

        #load the dataset
        if 0 in args.v:
            #master dataset
            master_path = os.path.join(data_root_dir,"dataset") #path to master dataset
            add_patch(master_path)

        #add patches here
        if 1 in args.v:
            #Class 10 - Ketchup
            class10_path = os.path.join(data_root_dir,"ketchup")
            add_patch(class10_path,patch_annotations_path = "labels",
                    patch_train_annotations_name="train_ketchup_annotations.json",
                    patch_val_annotations_name="val_ketchup_annotations.json")

            #Class 12 - Nestle All Purpose Cream
            class12_path = os.path.join(data_root_dir,"12_nestle_all_purpose_cream")
            add_patch(class12_path,patch_annotations_path = "labels",
                    patch_train_annotations_name="train_nestle.json",
                    patch_val_annotations_name="val_nestle.json")

            #Class A - Additional Images
        if 2 in args.v:
            classA_path = os.path.join(data_root_dir,"dataset_v3")
            add_patch(classA_path)

        if 3 in args.v:
            #Class A21 - Additional Soy Sauce Images
            classA21_path = os.path.join(data_root_dir,"silverswan")
            add_patch(classA21_path)


        #Combine annotations to a single directory
        def get_annotations(annotation_dirs,new_annotation_dir,category_dir="",ex_id = 100000):
            new_annotations = {"images":[],"annotations":[],"categories":[]}
            ids = {"images":set(),"annotations":set(),"categories":set()}
            for annotation_dir in annotation_dirs:
                add_categories = category_dir == annotation_dir if category_dir else annotation_dir
                with open(annotation_dir,"r",encoding="utf-8") as f:
                    annotations = json.load(f)

                    #Some ids don't match the filename. This segment corrects the issue.
                    true_ids = {}
                    false_ids = set()
                    for column in ["images","annotations"]:
                        for i in range(len(annotations[column])):
                            if column == "images":
                                if annotations[column][i]["id"] != int(os.path.splitext(annotations[column][i]["file_name"])[0]):
                                    #print(f"Replacing image id {annotations[column][i]["id"]} with {int(os.path.splitext(annotations[column][i]["file_name"])[0])}")
                                    true_ids[annotations[column][i]["id"]] = int(os.path.splitext(annotations[column][i]["file_name"])[0])
                                    false_ids.add(annotations[column][i]["id"])
                                    annotations[column][i]["id"] = int(os.path.splitext(annotations[column][i]["file_name"])[0])
                            if column == "annotations": #all image ids should already be corrected
                                if annotations[column][i]["image_id"] in false_ids:
                                    #print(f"Replacing annotation image id {annotations[column][i]["image_id"]} with {true_ids.get(annotations[column][i]["image_id"],annotations[column][i]["image_id"])}")
                                    annotations[column][i]["image_id"] = true_ids.get(annotations[column][i]["image_id"],annotations[column][i]["image_id"])

                    for column in ["images","annotations","categories"] if add_categories else ["images","annotations"]:
                        for i in range(len(annotations[column])):
                            if annotations[column][i]["id"] not in ids[column]:
                                ids[column].add(annotations[column][i]["id"])
                                new_annotations[column].append(annotations[column][i])
                            else:
                                print(f"WARNING: CONFLICTING ID {annotations[column][i]["id"]} FOR {column} IN {annotation_dir}!")
                                #Conflicting annotation ids can simply be replaced. Conveniently, none of them exceed 999999.
                                if column == "annotations":
                                    annotations[column][i]["id"] = ex_id
                                    ids[column].add(annotations[column][i]["id"])
                                    new_annotations[column].append(annotations[column][i])
                                    print(f"CONFLICTING ANNOTATION ID RESOLVED TO {new_annotations[column][-1]["id"]} == {ex_id}")
                                    ex_id += 1
                                #Skip conflicting image ids as they will be copied to the correct location anyway

            with open(new_annotation_dir,"w",encoding="utf-8") as f:
                json.dump(new_annotations,f)
            return ex_id

        ex_id = 100000 #use this to resolve conflicting image/annotation ids across multiple datasets
        #Use the additional dataset for categories
        category_train_dir = os.path.join(classA_path,"annotations","instances_train.json")
        category_val_dir = os.path.join(classA_path,"annotations","instances_val.json")
        #make a local copy of the annotations
        print("Compiling training annotations...")
        if not os.path.isdir(new_train_annotation_dir):
            os.makedirs(new_annotation_dir,exist_ok=True)
            ex_id = get_annotations(train_annotation_dirs,new_train_annotation_dir,category_train_dir,ex_id)
        print("Compiling validation annotations...")
        if not os.path.isdir(new_val_annotation_dir):
            os.makedirs(new_annotation_dir,exist_ok=True)
            ex_id = get_annotations(val_annotation_dirs,new_val_annotation_dir,category_val_dir,ex_id)
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


        #apply pre-processed patches
        #Warning: no conflict checking will take place here!
        if 4 in args.v:
            #Patch in the additional Nestle data
            class12p_path = os.path.join(data_root_dir,"Additional Data AP Cream")
            for add_image in os.listdir(os.path.join(class12p_path,"images")):
                shutil.copy2(os.path.join(class12p_path,"images",add_image),os.path.join(new_train_dir,add_image))
            for add_train_ann in os.listdir(os.path.join(class12p_path,"labels-segment")):
                with open(os.path.join(class12p_path,"labels-segment",add_train_ann),"r",encoding="utf-8") as f:
                    with open(os.path.join(new_data_dir,"yolo_annotations","labels","train",add_train_ann),"w",encoding="utf-8") as g:
                        g.writelines(f.readlines()+["\n"]) #needed to separate the annotations
            for add_train_box in os.listdir(os.path.join(class12p_path,"labels- detect")):
                with open(os.path.join(class12p_path,"labels- detect",add_train_box),"r",encoding="utf-8") as f:
                    with open(os.path.join(new_data_dir,"yolo_annotations","labels","train",add_train_box),"a",encoding="utf-8") as g:
                        g.writelines(f.readlines())
            print("Patched in additional Class 12 data!")

        if 5 in args.v:
            #Patch in the additional low light data
            class12l_path = os.path.join(data_root_dir,"coco_converted19")
            for add_image in os.listdir(os.path.join(class12l_path,"images")):
                shutil.copy2(os.path.join(class12l_path,"images",add_image),os.path.join(new_train_dir,add_image))
            for add_train_ann in os.listdir(os.path.join(class12l_path,"labels","train")):
                with open(os.path.join(class12l_path,"labels","train",add_train_ann),"r",encoding="utf-8") as f:
                    with open(os.path.join(new_data_dir,"yolo_annotations","labels","train",add_train_ann),"w",encoding="utf-8") as g:
                        g.writelines(f.readlines())
            print("Patched in additional low light data!")

        if 6 in args.v:
            #Patch in the additional Del Monte data
            class16p_path = os.path.join(data_root_dir,"additional_pineapple")
            for add_image in os.listdir(os.path.join(class16p_path,"images")):
                shutil.copy2(os.path.join(class16p_path,"images",add_image),os.path.join(new_train_dir,add_image))
            for add_train_ann in os.listdir(os.path.join(class16p_path,"pineapple-additional_segment")):
                with open(os.path.join(class16p_path,"pineapple-additional_segment",add_train_ann),"r",encoding="utf-8") as f:
                    with open(os.path.join(new_data_dir,"yolo_annotations","labels","train",add_train_ann),"w",encoding="utf-8") as g:
                        g.writelines(f.readlines()+["\n"]) #needed to separate the annotations
            for add_train_box in os.listdir(os.path.join(class16p_path,"pineapple-additional_detect")):
                with open(os.path.join(class16p_path,"pineapple-additional_detect",add_train_box),"r",encoding="utf-8") as f:
                    with open(os.path.join(new_data_dir,"yolo_annotations","labels","train",add_train_box),"a",encoding="utf-8") as g:
                        g.writelines(f.readlines())
            print("Patched in additional Class 12 data!")

        os.makedirs(os.path.join(new_data_dir,"labels","train"),exist_ok=True)
        for ann in os.listdir(os.path.join(new_data_dir,"yolo_annotations","labels","train")):
            shutil.copy2(os.path.join(new_data_dir,"yolo_annotations","labels","train",ann),os.path.join(new_data_dir,"labels","train",ann))
        os.makedirs(os.path.join(new_data_dir,"labels","val"),exist_ok=True)
        for ann in os.listdir(os.path.join(new_data_dir,"yolo_annotations","labels","val")):
            shutil.copy2(os.path.join(new_data_dir,"yolo_annotations","labels","val",ann),os.path.join(new_data_dir,"labels","val",ann))
        shutil.rmtree(os.path.join(new_data_dir,"yolo_annotations"))
        print("Converted COCO dataset to YOLO format.")


        labeladd_path = os.path.join(data_root_dir,"labels")
        #Apply fixed dataset points and overwrite old ones. Must be in train/val subfolders.
        if os.path.isdir(os.path.join(labeladd_path,"train")):
            for ann in os.listdir(os.path.join(labeladd_path,"train")):
                shutil.copy2(os.path.join(labeladd_path,"train",ann),os.path.join(new_data_dir,"labels","train",ann))
                #Fix incorrect annotations
                label_corrected = False
                with open(os.path.join(new_data_dir,"labels","train",ann),"r",encoding="utf-8") as f:
                    text = f.read()
                    if re.search("[^0-9. \r\n]+",text):
                        fixed_text = re.sub("[^0-9. \r\n]+","",text)
                        print(f"Annotation {ann} has incorrect annotations!")
                        print(f"Incorrect: {text}")
                        print(f"Correct: {fixed_text}")
                        label_corrected = True
                if label_corrected:
                    with open(os.path.join(new_data_dir,"labels","train",ann),"w",encoding="utf-8") as f:
                        f.write(fixed_text)
                        print(f"Fixed annotation {ann}")
            print("Added fixed train labels.")
        if os.path.isdir(os.path.join(labeladd_path,"val")):
            for ann in os.listdir(os.path.join(labeladd_path,"val")):
                shutil.copy2(os.path.join(labeladd_path,"val",ann),os.path.join(new_data_dir,"labels","val",ann))
                #Fix incorrect annotations
                label_corrected = False
                with open(os.path.join(new_data_dir,"labels","val",ann),"r",encoding="utf-8") as f:
                    text = f.read()
                    if re.search("[^0-9. \r\n]+",text):
                        fixed_text = re.sub("[^0-9. \r\n]+","",text)
                        print(f"Annotation {ann} has incorrect annotations!")
                        print(f"Incorrect: {text}")
                        print(f"Correct: {fixed_text}")
                        label_corrected = True
                if label_corrected:
                    with open(os.path.join(new_data_dir,"labels","val",ann),"w",encoding="utf-8") as f:
                        f.write(fixed_text)
                        print(f"Fixed annotation {ann}")
            print("Added fixed val labels.")
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
            if isinstance(custom_categories,dict) and custom_categories:
                return custom_categories
            categories = {}
            #check if the categories start at index 0 and adjust annotations if not
            offset_categories = 1 if all([int(annotations["categories"][i]["id"]) for i in range(len(annotations["categories"]))]) else 0
            for i in range(len(annotations["categories"])):
                categories[int(annotations["categories"][i]["id"])-offset_categories] = annotations["categories"][i]["name"]
            categories = [categories.get(i,f"CLASS_{i}_") for i in range(max(categories.keys())+offset_categories)]
            print(categories)
            return categories

        custom_categories = {
        0: "Coke Zero Bottled",
        1: "Eden Cheese",
        2: "KitKat",
        3: "Nescafe 3-in-1 Twin Pack",
        4: "Alaska Classic 377g Can",
        5: "Simply Pure Canola Oil",
        6: "Purefoods Corned Beef",
        7: "Whole Bulb of Garlic",
        8: "Lucky Me Pansit Canton",
        9: "UFC Banana Ketchup",
        10: "Whole Lemon",
        11: "Nestle All Purpose Cream 250ml",
        12: "Lady's Choice Real Mayonnaise 220 ml jar",
        13: "Skippy Peanut Butter",
        14: "Royal Pasta",
        15: "Del Monte Pineapple Juice",
        16: "Rebisco Crackers",
        17: "555 Sardines",
        18: "Sunsilk Shampoo",
        19: "Dove Lavender Soap",
        20: "Silver Swan Soy Sauce - 385 mL",
        21: "Colgate (Advanced White) Value Pack (2 Tubes)",
        22: "Century Tuna",
        23: "GreenCross Alcohol",
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


    os.makedirs(model_dir,exist_ok=True)
    models = {f"YOLOv11 Segmentation {mc.upper()}":f"yolo11{mc.lower()}-seg.pt" for mc in args.b}
    #Train the model(s) on the dataset
    num_epochs = args.n
    devices = args.d
    if -1 in devices:
        devices = "cpu"
    else:
        devices = list(filter(lambda x: x in DEVICE_LIST,devices))
        if len(devices) == 1:
            devices = devices[0]
        elif len(devices) == 0:
            devices = 0 if torch.cuda.is_available() else "cpu"


    print(f"Using device: {devices}")
    print(f"Training for {num_epochs} epochs")

    train_settings = {
        "data":yaml_dir,
        #Uncomment the "save" and "save_period" settings to save each iteration. This will take up more disk space, so beware!
        "save":True,
        "save_period":10,
        #comment "cache" if you don't have enough memory
        "cache":True,
        "project":model_dir,
        "exist_ok":True,
        "val":True,
        "plots":True,
        "device":devices,
        "epochs":num_epochs,
        "seed":42,
        "imgsz":640,
        #optimization settings
        "optimizer":"auto",
        "dropout":0.05,
        "close_mosaic":20, #max(1,int(num_epochs*0.02)), #last 2% of training without mosaic to learn whole shapes
        "verbose":True,
        "patience":100,
    }
    if args.a:
        augment_settings = {
            "hsv_h": 0.02,
            "hsv_v": 0.7,
            "degrees": 180,
            "shear": 45,
            "flipud": 0.25,
            "mixup": 0.2,
            "copy_paste": 0.2,
        }
        train_settings.update(augment_settings)
    if args.l:
        print(f"Transfer learning using model {args.l}")
        #check if path is valid
        if os.path.isfile(args.l) and os.path.splitext(args.l)[1] == ".pt":
            pretrained = args.l
        elif os.path.isdir(args.l):
            #prioritize using best weights
            if os.path.isfile(os.path.join(args.l,"weights","best.pt")):
                pretrained = os.path.join(args.l,"weights","best.pt")
            elif os.path.isfile(os.path.join(args.l,"weights","last.pt")):
                pretrained = os.path.join(args.l,"weights","last.pt")
            else:
                pretrained = ""
        else:
            pretrained = ""
        if pretrained:
            print(f"Using transfer learning with backbone {pretrained}")
            transfer_settings = {
                "pretrained":pretrained,
                "freeze":4,
            }
            train_settings.update(transfer_settings)

    if args.w:
        train_settings["workers"] = 0
        
    #Perform training
    if args.t:

        for k,v in models.items():
            model_name = k+(" (Augmented)" if args.a else "")+("_Transfer" if args.l else "")
            print(f"{model_name}: {os.path.join(model_dir,v)}")
            torch.cuda.empty_cache()
            #reload last checkpoint if exists
            if os.path.isfile(os.path.join(model_dir,model_name,"weights","last.pt")):
                print(f"Reloading model {model_name}!")
                model = YOLO(os.path.join(model_dir,model_name,"weights","last.pt"),task="segment")
            else:
                print(f"Training new model {model_name}!")
                model = YOLO(v,task="segment")

            #Override devices and batch size
            #Try batch sizes from 128*GPU to 1 in intervals of log2
            for batch_size in args.s:
                train_settings["batch"] = batch_size if not isinstance(devices,(list,set,tuple)) else batch_size*len(devices)
                print(f"Training with batch size {batch_size}->{train_settings["batch"]} over {1 if not isinstance(devices,(list,set,tuple)) else len(devices)} devices.")
                try:
                    if os.path.isfile(os.path.join(model_dir,model_name,"weights","last.pt")):
                        trained_model = model.train(
                            batch=train_settings["batch"],
                            device=train_settings["device"],
                            epochs=train_settings["epochs"],
                            patience=train_settings["patience"],
                            close_mosaic=train_settings["close_mosaic"],
                            resume=True,
                            )
                    else:
                        trained_model = model.train(
                            name=model_name,
                            **train_settings,
                            )
                except torch.OutOfMemoryError as e:
                    print(f"{model_name} failed to train! Error: {e}")
                    print("Retrying with smaller batch size...")
                    continue
                except AssertionError as e:
                    print(f"{model_name} failed to train! Error: {e}")
                    print("Retrying with smaller batch size...")
                    continue
                except Exception as e:
                    print(f"{model_name} failed to train! Error: {e}")
                    print("Retrying with smaller batch size...")
                    continue
                else:
                    print(f"Training of {k} complete!")
                    break
            model.data = None
            del model
            torch.cuda.empty_cache()

    #Validate all models
    for k,v in models.items():
        model_name = k+(" (Augmented)" if args.a else "")+("_Transfer" if args.l else "")
        try:
            model = YOLO(os.path.join(model_dir,model_name,"weights","last.pt"),task="segment")
            torch.cuda.empty_cache()
            for batch_size in args.s:
                train_settings_batch = batch_size if not isinstance(devices,(list,set,tuple)) else batch_size*len(devices)
                print(f"Validating with batch size {batch_size}->{train_settings_batch} over {1 if not isinstance(devices,(list,set,tuple)) else len(devices)} devices.")
                try:
                    model.val(save_json=True,
                        plots=True,
                        save_hybrid=True,
                        device=devices,
                        batch=train_settings_batch,
                        project=model_dir,
                        name=model_name+"-last")
                except Exception as e:
                    print(f"{model_name} failed to validate! Error: {e}")
                    print("Retrying with smaller batch size...")
                    continue
                else:
                    print(f"Validation of {k} complete!")
                    break
            model.data = None
            del model
            torch.cuda.empty_cache()
            model = YOLO(os.path.join(model_dir,model_name,"weights","best.pt"),task="segment")
            for batch_size in args.s:
                train_settings_batch = batch_size if not isinstance(devices,(list,set,tuple)) else batch_size*len(devices)
                print(f"Validating with batch size {batch_size}->{train_settings_batch} over {1 if not isinstance(devices,(list,set,tuple)) else len(devices)} devices.")
                try:
                    model.val(save_json=True,
                        plots=True,
                        save_hybrid=True,
                        device=devices,
                        batch=train_settings_batch,
                        project=model_dir,
                        name=model_name+"-best")
                except Exception as e:
                    print(f"{model_name} failed to validate! Error: {e}")
                    print("Retrying with smaller batch size...")
                    continue
                else:
                    print(f"Validation of {k} complete!")
                    break
            model.data = None
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{k} failed to validate! Error: {e}")
            continue
