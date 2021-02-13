# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:45:28 2020

@author: edgou
"""

import pandas as pd
import numpy as np
import pickle as pk
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split

import sys
import os

max_frame=10000
num_joint=33
max_body=2
toolbar_width = 30

code_emotions = ['CO','JO','NE','PE','TR']
file = "C://Users/llx33/Desktop/dataset2/data_PBME/data_PBME/ARJAWITRE05.csv"
data_path = 'C://Users/llx33/Desktop/dataset2/data_PBME/data_PBME'
out_path = 'test/Data_Lille'

benchmark = '80-20'

#%%
  
def csv_to_numpy_10(file):
    """Convert the csv file into an array numpy of size (NumJoint-8, 3, NumFrame). Delete joints 1,2,3,4,7,8,11,28,30,13.
    File must be of type string"""
    df = pd.read_csv(file, sep=',')
    #Initialisation du dataset
    delete_joints = [1,2,3,4,7,8,11,28,30,13]
    colnames = []
    for col in df:   
        colnames.append(col)
    #Implémentation des 3 première colonnes
    data_set = np.array([[df[colnames[0]],df[colnames[1]],df[colnames[2]]]])
    #Ajout des colonnes restantes
    for indice in range (5, len(colnames)+1, 3) :
        if indice//3 not in delete_joints:
            data = np.array([[df[colnames[indice-2]],df[colnames[indice-1]],df[colnames[indice]]]])
            data_set = np.concatenate((data_set,data),axis=0)
    return data_set
    
def csv_to_numpy(file):
    """Convert the csv file into an array numpy of size (NumJoint, 3, NumFrame).
    File must be of type string"""
    df = pd.read_csv(file, sep=',')
    #Initialisation du dataset
    colnames = []
    for col in df:  
        colnames.append(col)
    #Implémentation des 3 première colonnes
    data_set = np.array([[df[colnames[0]],df[colnames[1]],df[colnames[2]]]])
    #Ajout des colonnes restantes
    for indice in range (5, len(colnames)+1, 3) :
        data = np.array([[df[colnames[indice-2]],df[colnames[indice-1]],df[colnames[indice]]]])
        data_set = np.concatenate((data_set,data),axis=0)
    return data_set


def read_xyz(file, max_body=2, num_joint=33):
    """Adapt the array given by csv_to_numpy in an array of good shape to feed the neural network.
    File must be of type string""" 
    #Initialisation des variables
    seq_info = csv_to_numpy_10(file)
    data = np.zeros((3, seq_info.shape[2], num_joint, max_body))
    #Parcours de la seq_info pour récupérer les données
    for frame in range (seq_info.shape[2]):
        for joint in range(num_joint):
                    data[:, frame, joint, 0] = [seq_info[joint, 0, frame], seq_info[joint, 1, frame], seq_info[joint, 2, frame]]
    return data


def name_to_label(filename):
    """Associate a label to a video"""
    label = -1
    #Code for the label in the filename
    emo = filename[6:8]
    for k in range (len(code_emotions)):
        if emo == code_emotions[k]:
            label = k
    return label


def print_toolbar(rate, annotation=''):
    """Print the processing of dataset creation"""
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")

#%%
    
def split_dataset_random(data_path, test_size=0.2):
    """Split the dataset in two sets train and test with a repartition given by test_size
    data_path is a string
    test_size is a float between 0 and 1"""
    #Initialise list for the names of the files
    sample_name = []
    #Get the file of the dataset
    for filename in os.listdir(data_path):
        sample_name.append(filename)
    #Split the dataset in train and test with 80% in train and 20% in test
    train_sample, test_sample = train_test_split(sample_name, test_size=test_size)
    return (train_sample, test_sample)


def gendata_lille(data_path, out_path, data_sample, benchmark, part):
    """Generate the data to feed the neural network.
    Create a pkl file which associate the filename from data_path to its label and a npy file which contains the whole data end the position of the joints.
    Save the files created in out_path.
    data_path is a string
    out_path is a string
    data_sample is a list
    benchmark is a string
    part is a string"""
    #Initialise the list of label for the data
    data_label = []
    for filename in data_sample:
        data_label.append(name_to_label(filename)) #Label
    #Create a directory if out_path does not still exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #Create a pkl file which associate the name of a file with its label
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pk.dump((data_sample, list(data_label)), f)
    #Create a npy file containing the whole data to feed the neural network
    fp = open_memmap('{}/{}_data.npy'.format(out_path, part), dtype='float32', mode='w+', shape=(len(data_label), 3, max_frame, num_joint, max_body))
    for i, s in enumerate(data_sample):
        print_toolbar(i * 1.0 / len(data_label), '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format( i + 1, len(data_sample), benchmark, part))
        data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
        end_toolbar()


   
def split_dataset_sub(data_path, test_size=0.2):
    """Split the dataset in two sets train and test with distinct subjects in each dataset.
    data_path is a string
    test_size is a float between 0 and 1"""
    #Initialise list for the names of the files
    sample_name = []
    train_sample, test_sample = [], []
    subjects = []
    #Get the file of the dataset
    for filename in os.listdir(data_path):
        sample_name.append(filename)
        #Get the code subject code
        sub = filename[2:6]
        #Create a list with all the different subject code
        if sub not in subjects:
            subjects.append(sub)
    #Divide the dataset by taking 20% subjects in test and 80% others in train
    test_length = round(len(subjects)*test_size)
    for filename in sample_name:
        if filename[2:6] in subjects[:test_length]:
            test_sample.append(filename)
        else:
            train_sample.append(filename)
    return (train_sample, test_sample)


#%%

train, test = split_dataset_sub(data_path)
gendata_lille(data_path, out_path, train, benchmark, 'train')
gendata_lille(data_path, out_path, test, benchmark, 'test')

        
#%%
