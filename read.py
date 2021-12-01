import os

def read_data(data_path, labels_path):
    # get labels
    labels = open(labels_path, 'r').readlines()
    #should be 150 for facedata\facedatatestlabels, should be 1000 for digitdata/testdata
    #print(len(labels))

    #get data. f contains all lines
    f = open(data_path, 'r').readlines()
    height = int(len(f)/len(labels))  #height of each image = total num of lines / number of labels
    #print(height)

    #data will be a 3D array of the faces
    data = []
    for i in range(len(labels)):
        data.append([])
        #we remove the newline char at the end (makes printing nicer) and then call list() on it to turn it into a char array
        for j in range(height):
            data[i].append(list(f[i*height+j][:-1]))
 
    return data, labels

def print_entry(data, pos):
    for i in data[pos]:
        print(''.join(i))

def print_dataset(data, percent):
    for i in range(int(len(data) * percent)):
        print_entry(data, i)
        print("-----------------------------------------------------------")

def read_datasets(DIGITS_DATA_PATH, DIGITS_LABEL_PATH, FACES_DATA_PATH, FACES_LABEL_PATH):
    #change paths to whatever you need it to be
    

    digits, digits_labels = read_data(DIGITS_DATA_PATH, DIGITS_LABEL_PATH)
    faces, faces_labels = read_data(FACES_DATA_PATH, FACES_LABEL_PATH)

    #print_entry(numbers, 72)
    #print_entry(faces, 35)
    #print_dataset(faces, 0.1)

    #dataset can be updated to store more information than just the raw data and their correct labels
    digits_dataset = [digits, digits_labels]
    faces_dataset = [faces, faces_labels]

    return digits_dataset, faces_dataset

