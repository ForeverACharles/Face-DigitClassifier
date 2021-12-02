from read import read_dataset, print_entry, print_dataset
from perceptron import p_digits_train, p_digits_evaluate, p_faces_train, p_faces_evaluate
from naive_bayes import nb_digits_train, nb_digits_evaluate, nb_faces_train, nb_faces_evaluate
from custom_algo import ca_digits_train, ca_digits_evaluate, ca_faces_train, ca_faces_evaluate
import time
import os
import math
import numpy

def main():

    digits_dataset, faces_dataset = read_datasets()

    perceptron = [p_digits_train, p_digits_evaluate, p_faces_train, p_faces_evaluate]
    naive_bayes = [nb_digits_train, nb_digits_evaluate, nb_faces_train, nb_faces_evaluate]
    custom_algo = [ca_digits_train, ca_digits_evaluate, ca_faces_train, ca_faces_evaluate]

    #add or remove algorithms to test with
    algorithms = [perceptron, naive_bayes, custom_algo]

    #run 10% up to end_percent iterations of each algorithm
    end_percent = 0.5
    summary = run_and_report(algorithms, digits_dataset, faces_dataset, end_percent)
    report_summary(summary)

def read_datasets():
    curr_dir = ""
    if(not os.path.isdir("data")):
        curr_dir = os.path.dirname(__file__) + "\\"

    DIGITS_TEST_DATA = curr_dir + "data\\digitdata\\testimages"
    DIGITS_TEST_LABELS = curr_dir + "data\\digitdata\\testlabels"
    DIGITS_TRAIN_DATA =  curr_dir + "data\\digitdata\\trainingimages"
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

def run_and_report(algorithms, digits_dataset, faces_dataset, end_percent):

    summary = []
    percent = 0.1

    while percent <= end_percent:

        print("Training on " + str(round(percent * 100, 2)) + "% of training data")
        digits_results, faces_results = run_all(algorithms, digits_dataset, faces_dataset, percent)

        summary.append(report_results(digits_results, faces_results, percent))
        percent = round(percent + 0.1, 2)

    return summary
    
def run_all(algorithms, digits_dataset, faces_dataset, percent):

    digits_results = []
    faces_results = []

    for algorithm in algorithms:
        digits_train, digits_evaluate = algorithm[0], algorithm[1]
        train_time, result = run_algo(digits_dataset, percent, digits_train, digits_evaluate)
        digits_results.append([train_time, result])

        faces_train, faces_evaluate = algorithm[2], algorithm[3]
        train_time, result = run_algo(faces_dataset, percent, faces_train, faces_evaluate)
        faces_results.append([train_time, result])

    return digits_results, faces_results

def run_algo(dataset, percent, train, evaluate):
    train_dataset = trim_dataset(dataset[0], percent)
    data = dataset[1]

    start = time.time()
    weights = train(train_dataset)
    end = time.time()

    correct, wrong = evaluate(data, weights)

    return round((end - start), 3), [correct, wrong]

def report_results(digits_results, faces_results, percent):
    a = 0
    algorithms = ["Perceptron", "Naive Bayes", "Custom Algo"]
    print("\n" + str(round(percent * 100, 2)) + "% Training Data Results:")

    report = [[],[]]

    print('===============DIGITS===============')
    for digits_result in digits_results:
        report[0].append(digits_result)

        print(algorithms[a % 3])
        print("Training took "+ str(digits_result[0]) + " seconds")
        print("successes: "+ str(digits_result[1][0]) + " failures: "+ str(digits_result[1][1]) +"\n")
        a+=1
    
    print('===============FACES===============')
    for faces_result in faces_results:
        report[1].append(faces_result)

        print(algorithms[a % 3])
        print("Training took "+ str(faces_result[0]) + " seconds")
        print("successes: "+ str(faces_result[1][0]) + " failures: "+ str(faces_result[1][1]) +"\n")
        a+=1

    return report

def report_summary(summary):

    print("\nReporting Summary:")
    print("-----------------------Digits----------------------    -----------------------Faces-----------------------")
    print("Train\tPerceptron\tNaive Bayes\tCustom Algo\tTrain\tPerceptron\tNaive Bayes\tCustom Algo")
    for i in range(len(summary)):
        print(str(round(0.1 * (i + 1) * 100, 2)) + "%", end="")
        print("\t" + str(summary[i][0][0][0]) + " sec\t" + str(summary[i][0][1][0]) + " sec\t\t" + str(summary[i][0][2][0]) + " sec", end="\t\t")
        print(str(round(0.1 * (i + 1) * 100, 2)) + "%", end="")
        print("\t" + str(summary[i][1][0][0]) + " sec\t" + str(summary[i][1][1][0]) + " sec\t\t" + str(summary[i][1][2][0]) + " sec")

        print("\t" + str(summary[i][0][0][1][0]) + " right\t" + str(summary[i][0][1][1][0]) + " right\t\t" + str(summary[i][0][2][1][0]) + " right", end="\t\t")
        print("\t" + str(summary[i][1][0][1][0]) + " right\t" + str(summary[i][1][1][1][0]) + " right\t\t" + str(summary[i][1][2][1][0]) + " right")

        print("\t" + str(summary[i][0][0][1][1]) + " wrong\t" + str(summary[i][0][1][1][1]) + " wrong\t\t" + str(summary[i][0][2][1][1]) + " wrong", end="\t\t")
        print("\t" + str(summary[i][1][0][1][1]) + " wrong\t" + str(summary[i][1][1][1][1]) + " wrong\t\t" + str(summary[i][1][2][1][1]) + " wrong\n")

    digits_means, faces_means = get_means(summary)
    print("Avg.\t" + str(digits_means[0][0]) + " sec\t" + str(digits_means[0][1]) + " sec\t\t" + str(digits_means[0][2]) + " sec", end="\t\t")
    print("Avg.\t" + str(faces_means[0][0]) + " sec\t" + str(faces_means[0][1]) + " sec\t\t" + str(faces_means[0][2]) + " sec")

    print("Avg.\t" + str(digits_means[1][0]) + " right\t" + str(digits_means[1][1]) + " right\t" + str(digits_means[1][2]) + " right", end="\t")
    print("Avg.\t" + str(faces_means[1][0]) + " right\t" + str(faces_means[1][1]) + " right\t" + str(faces_means[1][2]) + " right")

    print("Avg.\t" + str(digits_means[2][0]) + " wrong\t" + str(digits_means[2][1]) + " wrong\t" + str(digits_means[2][2]) + " wrong", end="\t")
    print("Avg.\t" + str(faces_means[2][0]) + " wrong\t" + str(faces_means[2][1]) + " wrong\t" + str(faces_means[2][2]) + " wrong")

    print("\n")

def get_means(summary):
    digits_time_sum = [0, 0, 0]
    digits_right_sum = [0, 0, 0]
    digits_wrong_sum = [0, 0, 0]
    faces_time_sum = [0, 0, 0]
    faces_right_sum = [0, 0, 0]
    faces_wrong_sum = [0, 0, 0]

    for i in range(len(summary)):
        for a in range(3):
            digits_time_sum[a] += summary[i][0][a][0]
            digits_right_sum[a] += summary[i][0][a][1][0]
            digits_wrong_sum[a] += summary[i][0][a][1][1]

            faces_time_sum[a] += summary[i][1][a][0]
            faces_right_sum[a] += summary[i][1][a][1][0]
            faces_wrong_sum[a] += summary[i][1][a][1][1]

    digits_time_mean = numpy.divide(digits_time_sum, len(summary))
    digits_right_mean = numpy.divide(digits_right_sum, len(summary))
    digits_wrong_mean = numpy.divide(digits_wrong_sum, len(summary))
    faces_time_mean = numpy.divide(faces_time_sum, len(summary))
    faces_right_mean = numpy.divide(faces_right_sum, len(summary))
    faces_wrong_mean = numpy.divide(faces_wrong_sum, len(summary))

    digits_means = numpy.round([digits_time_mean, digits_right_mean, digits_wrong_mean], 3)
    faces_means = numpy.round([faces_time_mean, faces_right_mean, faces_wrong_mean], 3)

    return digits_means, faces_means

#our testing will have us limit the percent of the dataset to perform training on
def trim_dataset(dataset, percent):

    trimmed_dataset = list.copy(dataset)
    size = int(len(dataset[0]) * round(percent, 3))
    for i in range(len(dataset)):
        trimmed_dataset[i] = dataset[i][:size]

    return trimmed_dataset

if __name__ == '__main__':
    main()