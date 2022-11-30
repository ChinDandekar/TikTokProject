#DataSplit.py
#Author: Chinmay Dandekar

import os
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
import csv
import json


def get_vids(path2ajpgs):
    listOfCats = os.listdir(path2ajpgs)
    ids = []
    labels = []
    for catg in listOfCats:
        if(catg != ".DS_Store" and catg != "val.csv" and catg != "test.csv" and catg != "train.csv" and catg != 'classids.json'):
            path2catg = os.path.join(path2ajpgs, catg)
            listOfSubCats = os.listdir(path2catg)
            path2subCats = [os.path.join(path2catg,los) for los in listOfSubCats]
            ids.extend(path2subCats)
            labels.extend([catg]*len(listOfSubCats))
        else:
            listOfCats.remove(catg)
    return ids, labels, listOfCats 

def create_label_dict(catgs):
    labels_dict = {}
    ind = 0
    for uc in catgs:
        labels_dict[uc] = ind
        ind+=1
    return labels_dict

def get_split(pathName, split):
    all_vids,all_labels,catgs = get_vids(pathName)

    labels_dict = create_label_dict(catgs)

    num_classes = len(labels_dict)

    unique_ids = [id_ for id_, label in zip(all_vids,all_labels) if labels_dict[label]<num_classes]
    unique_labels = [label for id_, label in zip(all_vids,all_labels) if labels_dict[label]<num_classes]

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    train_indx, testval_indx = next(sss.split(unique_ids, unique_labels))

    train_ids = [unique_ids[ind] for ind in train_indx]
    train_labels = [unique_labels[ind] for ind in train_indx]

    testval_ids = [unique_ids[ind] for ind in testval_indx]
    testval_labels = [unique_labels[ind] for ind in testval_indx]

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
    test_indx, val_indx = next(sss.split(testval_ids, testval_labels))

    test_ids = [testval_ids[ind] for ind in test_indx]
    test_labels = [testval_labels[ind] for ind in test_indx]

    val_ids = [testval_ids[ind] for ind in val_indx]
    val_labels = [testval_labels[ind] for ind in val_indx]

    assert split in ["train", "val", "test"]
    if split in ["train"]:
        return train_ids, train_labels, labels_dict
    elif split in ["val"]:
        return val_ids, val_labels, labels_dict
    else:
        return test_ids, test_labels, labels_dict

def create_transformer():
    h = 224
    w = 224
    transformer = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
        transforms.ToTensor(),
        ])
    return transformer

def write_csvs(path):
    train_ids, train_labels, labels_dict = get_split(path ,"train")
    val_ids, val_labels, labels_dict = get_split(path, "val")
    test_ids, test_labels, labels_dict = get_split(path, "test")

    pathToTrain = path + "train.csv"
    with open(pathToTrain, 'w', newline = '') as f:
        # create the csv writer
        writer = csv.writer(f)
        
        # write a row to the csv file
        for x in range(len(train_ids)):
            pair = [train_ids[x], labels_dict[train_labels[x]]]
            writer.writerow(pair)
    
    pathToTest = path + "test.csv"
    with open(pathToTest, 'w', newline = '') as f:
        # create the csv writer
        writer = csv.writer(f)
        
        # write a row to the csv file
        #for x in range(len(test_ids)):
        for x in range(19,20):
            pair = [test_ids[x], labels_dict[test_labels[x]]]
            writer.writerow(pair)
    
    pathToVal = path + "val.csv"
    with open(pathToVal, 'w', newline = '') as f:
        # create the csv writer
        writer = csv.writer(f)
        
        # write a row to the csv file
        for x in range(len(val_ids)):
            pair = [val_ids[x], labels_dict[val_labels[x]]]
            writer.writerow(pair)
    
    # Serializing json
    json_object = json.dumps(labels_dict)
    
    # Writing to sample.json
    pathToLabelDict = path + "classids.json"
    with open(pathToLabelDict, "w") as outfile:
        outfile.write(json_object)

