from read import read_dataset, print_entry, print_dataset
from perceptron import digits_train, digits_evaluate, faces_train, faces_evaluate
import time
import os
NICKS_COMPUTER = True
def main():
    if(NICKS_COMPUTER): 
        #curr_dir = os.path.dirname(__file__) + "\\"
        DIGITS_TEST_DATA =  "data\\digitdata\\testimages"
        DIGITS_TEST_LABELS = "data\\digitdata\\testlabels"
        DIGITS_TRAIN_DATA =  "data\\digitdata\\trainingimages"
        DIGITS_TRAIN_LABEL =   "data\\digitdata\\traininglabels"

        
        FACES_TEST_DATA = "data\\facedata\\facedatatest"
        FACES_TEST_LABEL =  "data\\facedata\\facedatatestlabels"
        FACES_TRAIN_DATA = "data\\facedata\\facedatatrain"
        FACES_TRAIN_LABEL = "data\\facedata\\facedatatrainlabels"
    else:
        curr_dir = os.path.dirname(__file__) + "\\"
        DIGITS_TEST_DATA = curr_dir+ "data\\digitdata\\testimages"
        DIGITS_TEST_LABELS = curr_dir+"data\\digitdata\\testlabels"
        FACES_TEST_DATA= curr_dir+"data\\facedata\\facedatatest"
        FACES_TEST_LABEL =  curr_dir+"data\\facedata\\facedatatestlabels"
        DIGITS_TRAIN_DATA =  curr_dir+"data\\digitdata\\trainingimages"
        DIGITS_TRAIN_LABEL =  curr_dir+ "data\\digitdata\\traininglabels"
    
    print('===============DIGITS===============')
    digits_train_dataset = read_dataset(DIGITS_TRAIN_DATA, DIGITS_TRAIN_LABEL)
    digits_dataset = read_dataset(DIGITS_TEST_DATA, DIGITS_TEST_LABELS)
    #digits_dataset, faces_dataset = read_dataset(DIGITS_TEST_DATA, DIGITS_TEST_LABELS, FACES_DATA_PATH, FACES_LABEL_PATH)
    #our testing will have us limit the percent of the dataset to perform training on
    digits_train_dataset = trim_dataset(digits_train_dataset, 1)
    #print_dataset(digits_dataset[0], 0.1)
    start = time.time()
    weights = digits_train(digits_train_dataset)
    end = time.time()
    print()
    print("digit training took "+str(end-start) + "seconds")
    print()

    digits_evaluate(digits_dataset, weights)
    print("\n\n\n")
    print('===============FACES===============')




    faces_train_dataset = read_dataset(FACES_TRAIN_DATA, FACES_TRAIN_LABEL)
    faces_dataset = read_dataset(FACES_TEST_DATA, FACES_TEST_LABEL)
    faces_train_dataset = trim_dataset(faces_train_dataset, 1)
    start = time.time()
    weights = faces_train(faces_train_dataset)
    end = time.time()
    print()
    print("faces training took "+str(end-start) + "seconds")
    print()
    faces_evaluate(faces_dataset, weights)

#our testing will have us limit the percent of the dataset to perform training on
def trim_dataset(dataset, percent):
    size = int(len(dataset[0]) * percent)
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:size]

    return dataset

if __name__ == '__main__':
    main()