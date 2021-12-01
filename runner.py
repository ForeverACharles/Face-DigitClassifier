from read import read_dataset, print_entry, print_dataset
from perceptron import p_train, p_evaluate
import time
import os
NICKS_COMPUTER = True
def main():
    if(NICKS_COMPUTER): 
        #curr_dir = os.path.dirname(__file__) + "\\"
        DIGITS_TEST_DATA =  "data\\digitdata\\testimages"
        DIGITS_TEST_LABELS = "data\\digitdata\\testlabels"
        FACES_DATA_PATH = "data\\facedata\\facedatatest"
        FACES_LABEL_PATH =  "data\\facedata\\facedatatestlabels"
        DIGITS_TRAIN_DATA =  "data\\digitdata\\trainingimages"
        DIGITS_TRAIN_LABEL =   "data\\digitdata\\traininglabels"
    else:
        curr_dir = os.path.dirname(__file__) + "\\"
        DIGITS_TEST_DATA = curr_dir+ "data\\digitdata\\testimages"
        DIGITS_TEST_LABELS = curr_dir+"data\\digitdata\\testlabels"
        FACES_DATA_PATH = curr_dir+"data\\facedata\\facedatatest"
        FACES_LABEL_PATH =  curr_dir+"data\\facedata\\facedatatestlabels"
        DIGITS_TRAIN_DATA =  curr_dir+"data\\digitdata\\trainingimages"
        DIGITS_TRAIN_LABEL =  curr_dir+ "data\\digitdata\\traininglabels"
    
    digits_train_dataset = read_dataset(DIGITS_TRAIN_DATA, DIGITS_TRAIN_LABEL)
    digits_dataset = read_dataset(DIGITS_TEST_DATA, DIGITS_TEST_LABELS)
    #digits_dataset, faces_dataset = read_dataset(DIGITS_TEST_DATA, DIGITS_TEST_LABELS, FACES_DATA_PATH, FACES_LABEL_PATH)
    #our testing will have us limit the percent of the dataset to perform training on
    digits_train_dataset = trim_dataset(digits_train_dataset, 0.1)
    #print_dataset(digits_dataset[0], 0.1)
    start = time.time()
    weights = p_train(digits_train_dataset)
    end = time.time()
    print()
    print("training took "+str(end-start) + "seconds")
    print()

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