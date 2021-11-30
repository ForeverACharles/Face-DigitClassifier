from read import read_dataset, print_entry, print_dataset
from perceptron import p_train, p_evaluate

import os

def main():
    curr_dir = os.path.dirname(__file__) + "\\"

    DIGITS_TRAIN_DATA =  curr_dir +  "data\\digitdata\\trainingimages"
    DIGITS_TRAIN_LABELS =   curr_dir +  "data\\digitdata\\traininglabels"
    DIGITS_TEST_DATA =  curr_dir + "data\\digitdata\\testimages"
    DIGITS_TEST_LABELS = curr_dir +  "data\\digitdata\\testlabels"
    
    FACES_TRAIN_DATA =  curr_dir + "data\\facedata\\facedatatrain"
    FACES_TRAIN_LABELS =  curr_dir +  "data\\facedata\\facedatatrainlabels"
    FACES_TEST_DATA =  curr_dir + "data\\facedata\\facedatatest"
    FACES_TEST_LABELS =  curr_dir +  "data\\facedata\\facedatatestlabels"
   

    digits_train_dataset = read_dataset(DIGITS_TRAIN_DATA, DIGITS_TRAIN_LABELS)
    digits_dataset = read_dataset(DIGITS_TEST_DATA, DIGITS_TEST_LABELS)

    digits_train_dataset = trim_dataset(digits_train_dataset, 0.1)
    digits_dataset = trim_dataset(digits_dataset, 0.1)
    
    #digits_dataset = trim_dataset(digits_dataset, 1)

    #print_dataset(digits_dataset[0], 0.1)

    weights = p_train(digits_train_dataset)
    p_evaluate(digits_dataset, weights)

    #p_train(faces_dataset)
    #p_evaluate(faces_dataset)

#our testing will have us limit the percent of the dataset to perform training on
def trim_dataset(dataset, percent):
    size = int(len(dataset[0]) * percent)
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:size]

    return dataset

if __name__ == '__main__':
    main()