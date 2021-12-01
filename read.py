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

#def read_datasets(DIGITS_DATA_PATH, DIGITS_LABEL_PATH, FACES_DATA_PATH, FACES_LABEL_PATH):
def read_dataset(DATA_PATH, LABEL_PATH):
    #change paths to whatever you need it to be
    

    data, data_labels = read_data(DATA_PATH, LABEL_PATH)
    #faces, faces_labels = read_data(FACES_DATA_PATH, FACES_LABEL_PATH)

    #print_entry(numbers, 72)
    #print_entry(faces, 35)
    #print_dataset(faces, 0.1)

    #dataset can be updated to store more information than just the raw data and their correct labels
    dataset = [data, data_labels]
    #digits_dataset = [digits, digits_labels]
    #faces_dataset = [faces, faces_labels]

    return dataset
