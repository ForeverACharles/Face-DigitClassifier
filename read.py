import os

def read_data(data_path, labels_path):

    print("Reading in dataset...")

    # get labels
    labels = open(labels_path, 'r').readlines()
    #should be 150 for facedata\facedatatestlabels, should be 1000 for digitdata/testdata
    
    #extract labels and turn them into ints 
    for i in range(len(labels)):
        labels[i] = int(labels[i])

    #get data. f contains all lines
    f = open(data_path, 'r').readlines()
    height = int(len(f)/len(labels))  #height of each image = total num of lines / number of labels
    print(height)

    #data will be a 3D array of the faces
    data = []
    for i in range(len(labels)):
        data.append([0])
        #we remove the newline char at the end (makes printing nicer) and then call list() on it to turn it into a char array
        for j in range(height):
            for k in range(len(f[i*height+j])-1):
                data[i].append(f[i*height+j][k])

    #preprocessing 
    for i in range(len(labels)):
        for j in range(len(data[0])):
                data[i][j] = 1 if data[i][j] != ' ' else 0
 
    return data, labels

def print_entry(data, pos):
    for i in data[pos]:
        print(''.join(i))

def print_dataset(data, percent):
    for i in range(int(len(data) * percent)):
        print_entry(data, i)
        print("-----------------------------------------------------------")

def read_dataset(DATA_PATH, LABEL_PATH):
 
    data, data_labels = read_data(DATA_PATH, LABEL_PATH)
    
    #dataset can be updated to store more information than just the raw data and their correct labels
    dataset = [data, data_labels]
    return dataset
