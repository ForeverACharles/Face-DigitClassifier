from read import read_datasets, print_entry, print_dataset
from perceptron import p_train, p_evaluate

import os

def main():
    #curr_dir = os.path.dirname(__file__) + "\\"
    DIGITS_TEST_DATA =  "data\\digitdata\\testimages"
    DIGITS_TEST_LABELS = "data\\digitdata\\testlabels"
    FACES_DATA_PATH = "data\\facedata\\facedatatest"
    FACES_LABEL_PATH =  "data\\facedata\\facedatatestlabels"
    DIGITS_TRAIN_DATA =  "data\\digitdata\\trainingimages"
    DIGITS_TRAIN_LABEL =   "data\\digitdata\\traininglabels"
    digits_train_dataset, faces_dataset = read_datasets(DIGITS_TRAIN_DATA, DIGITS_TRAIN_LABEL, FACES_DATA_PATH, FACES_LABEL_PATH)
    digits_dataset, faces_dataset = read_datasets(DIGITS_TEST_DATA, DIGITS_TEST_LABELS, FACES_DATA_PATH, FACES_LABEL_PATH)
    #our testing will have us limit the percent of the dataset to perform training on
    #digits_dataset = trim_dataset(digits_dataset, 1)

    #print_dataset(digits_dataset[0], 0.1)

    weights = p_train(digits_train_dataset)
    p_evaluate(digits_dataset, weights)

    #p_train(faces_dataset)
    #p_evaluate(faces_dataset)

def trim_dataset(dataset, percent):
    size = int(len(dataset[0]) * percent)
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:size]

    return dataset

if __name__ == '__main__':
    main()