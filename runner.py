from read import read_dataset, print_entry, print_dataset
from perceptron import p_digits_train, p_digits_evaluate, p_faces_train, p_faces_evaluate
from naive_bayes import nb_digits_train, nb_digits_evaluate, nb_faces_train, nb_faces_evaluate
from custom_algo import ca_digits_train, ca_digits_evaluate, ca_faces_train, ca_faces_evaluate
import time
import os

def main():

    digits_dataset, faces_dataset = read_datasets()

    perceptron = [p_digits_train, p_digits_evaluate, p_faces_train, p_faces_evaluate]
    naive_bayes = [nb_digits_train, nb_digits_evaluate, nb_faces_train, nb_faces_evaluate]
    custom_algo = [ca_digits_train, ca_faces_evaluate, ca_faces_train, ca_faces_evaluate]

    #add or remove algorithms to test with
    algorithmns = [perceptron, naive_bayes, custom_algo]

    percent = 1
    digits_results, faces_results = run_all(algorithmns, digits_dataset, faces_dataset, percent)
    report_results(digits_results, faces_results)

def read_datasets():
    curr_dir = ""
    if(not os.path.isdir("data")):
        curr_dir = os.path.dirname(__file__) + "\\"

    DIGITS_TEST_DATA = curr_dir + "data\\digitdata\\testimages"
    DIGITS_TEST_LABELS = curr_dir +"data\\digitdata\\testlabels"
    DIGITS_TRAIN_DATA =  curr_dir +"data\\digitdata\\trainingimages"
    DIGITS_TRAIN_LABEL =  curr_dir + "data\\digitdata\\traininglabels"

    FACES_TEST_DATA = curr_dir + "data\\facedata\\facedatatest"
    FACES_TEST_LABEL =  curr_dir + "data\\facedata\\facedatatestlabels"
    FACES_TRAIN_DATA = curr_dir + "data\\facedata\\facedatatrain"
    FACES_TRAIN_LABEL = curr_dir + "data\\facedata\\facedatatrainlabels"

    digits_train_dataset = read_dataset(DIGITS_TRAIN_DATA, DIGITS_TRAIN_LABEL)
    digits_dataset = read_dataset(DIGITS_TEST_DATA, DIGITS_TEST_LABELS)

    faces_train_dataset = read_dataset(FACES_TRAIN_DATA, FACES_TRAIN_LABEL)
    faces_dataset = read_dataset(FACES_TEST_DATA, FACES_TEST_LABEL)

    return [digits_train_dataset, digits_dataset], [faces_train_dataset, faces_dataset]

def run_all(algorithms, digits_dataset, faces_dataset, percent):

    digits_results = []
    faces_results = []

    for algorithm in algorithms:
        digits_train, digits_evaluate = algorithm[0], algorithm[1]
        train_time, result = run(digits_dataset, percent, digits_train, digits_evaluate)
        digits_results.append([train_time, result])

        faces_train, faces_evaluate = algorithm[2], algorithm[3]
        train_time, result = run(faces_dataset, percent, faces_train, faces_evaluate)
        faces_results.append([train_time, result])

    return digits_results, faces_results

def run(dataset, percent, train, evaluate):
    train_dataset = trim_dataset(dataset[0], percent)
    data = dataset[1]

    start = time.time()
    weights = train(train_dataset)
    end = time.time()

    correct, wrong = evaluate(data, weights)

    return round((end - start), 4), [correct, wrong]

def report_results(digits_results, faces_results):

    a = 0
    algorithms = ["Perceptron", "Naive Bayes", "Custom Algo"]

    print('\n===============DIGITS===============')
    for digits_result in digits_results:
        print(algorithms[a % 3])
        print("Training took "+ str(digits_result[0]) + " seconds")
        print("successes: "+ str(digits_result[1][0]) + " failures: "+ str(digits_result[1][1]) +"\n")
        a+=1
    
    print('===============FACES===============')
    for face_result in faces_results:
        print(algorithms[a % 3])
        print("Training took "+ str(face_result[0]) + " seconds")
        print("successes: "+ str(face_result[1][0]) + " failures: "+ str(face_result[1][1]) +"\n")
        a+=1

#our testing will have us limit the percent of the dataset to perform training on
def trim_dataset(dataset, percent):
    size = int(len(dataset[0]) * percent)
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:size]

    return dataset

if __name__ == '__main__':
    main()