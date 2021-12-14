from read import read_dataset, print_entry, print_dataset
from perceptron import p_digits_train, p_digits_evaluate, p_faces_train, p_faces_evaluate
from naive_bayes import nb_digits_train, nb_digits_evaluate, nb_faces_train, nb_faces_evaluate
from custom_algo import ca_digits_train, ca_digits_evaluate, ca_faces_train, ca_faces_evaluate
import time
import os
import math
import numpy
import random

def main():

    digits_dataset, faces_dataset = read_datasets()

    #add or remove algorithms to test with by passing in 1 or 0 for on and off
    algorithms = toggle_algorithms(1, 1, 1)

    #run up to end_percent of each algorithm (min of 0.1 and max of 1)
    end_percent = 0.1
    #number of iterations to do
    iterations = 2
    run_iterations(digits_dataset, faces_dataset, algorithms, end_percent, iterations)

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

def toggle_algorithms(p_toggle, nb_toggle, ca_toggle):

    perceptron = [do_nothing_train, do_nothing_eval, do_nothing_train, do_nothing_eval]
    naive_bayes = [do_nothing_train, do_nothing_eval, do_nothing_train, do_nothing_eval]
    custom_algo = [do_nothing_train, do_nothing_eval, do_nothing_train, do_nothing_eval]

    if p_toggle == 1:
        perceptron = [p_digits_train, p_digits_evaluate, p_faces_train, p_faces_evaluate]

    if nb_toggle == 1:
        naive_bayes = [nb_digits_train, nb_digits_evaluate, nb_faces_train, nb_faces_evaluate]

    if ca_toggle == 1:
        custom_algo = [ca_digits_train, ca_digits_evaluate, ca_faces_train, ca_faces_evaluate]

    algorithms = [perceptron, naive_bayes, custom_algo]
    return algorithms

def run_iterations(digits_dataset, faces_dataset, algorithms, end_percent, iterations):

    summaries = []
    for i in range(iterations):
        print("Running iteration: " + str(i + 1))
        summary = run_and_report(algorithms, digits_dataset, faces_dataset, end_percent)
        summaries.append(summary)
        report_summary(summary)

    p_stats, nb_stats, ca_stats = compile_statistics(summaries)
    report_statistics(p_stats, nb_stats, ca_stats, end_percent)
    
def compile_statistics(summaries):

    size = len(summaries[0])
    p_digits_runtimes = [[] for _ in range(size)]
    p_digits_rights = [[] for _ in range(size)]
    p_digits_wrongs = [[] for _ in range(size)]

    nb_digits_runtimes = [[] for _ in range(size)]
    nb_digits_rights = [[] for _ in range(size)]
    nb_digits_wrongs = [[] for _ in range(size)]

    ca_digits_runtimes = [[] for _ in range(size)]
    ca_digits_rights = [[] for _ in range(size)]
    ca_digits_wrongs = [[] for _ in range(size)]

    p_faces_runtimes = [[] for _ in range(size)]
    p_faces_rights = [[] for _ in range(size)]
    p_faces_wrongs = [[] for _ in range(size)]

    nb_faces_runtimes = [[] for _ in range(size)]
    nb_faces_rights = [[] for _ in range(size)]
    nb_faces_wrongs = [[] for _ in range(size)]

    ca_faces_runtimes = [[] for _ in range(size)]
    ca_faces_rights = [[] for _ in range(size)]
    ca_faces_wrongs = [[] for _ in range(size)]

    
    #report_summary(summary)
    for i in range(size):
        for summary in summaries:

            p_digits_runtimes[i].append(summary[i][0][0][0])
            p_digits_rights[i].append(summary[i][0][0][1][0])
            p_digits_wrongs[i].append(summary[i][0][0][1][1])

            nb_digits_runtimes[i].append(summary[i][0][1][0])
            nb_digits_rights[i].append(summary[i][0][1][1][0])
            nb_digits_wrongs[i].append(summary[i][0][1][1][1])

            ca_digits_runtimes[i].append(summary[i][0][2][0])
            ca_digits_rights[i].append(summary[i][0][2][1][0])
            ca_digits_wrongs[i].append(summary[i][0][2][1][1])

            p_faces_runtimes[i].append(summary[i][1][0][0])
            p_faces_rights[i].append(summary[i][1][0][1][0])
            p_faces_wrongs[i].append(summary[i][1][0][1][1])

            nb_faces_runtimes[i].append(summary[i][1][1][0])
            nb_faces_rights[i].append(summary[i][1][1][1][0])
            nb_faces_wrongs[i].append(summary[i][1][1][1][1])

            ca_faces_runtimes[i].append(summary[i][1][2][0])
            ca_faces_rights[i].append(summary[i][1][2][1][0])
            ca_faces_wrongs[i].append(summary[i][1][2][1][1])

    p_runtimes = [p_digits_runtimes, p_faces_runtimes]
    p_rights = [p_digits_rights, p_faces_rights]
    p_wrongs = [p_digits_wrongs, p_faces_wrongs]
    p_avgs = get_algo_avg(p_runtimes, p_rights, p_wrongs, size)
    p_stds = get_algo_std(p_runtimes, p_rights, p_wrongs, size)

    nb_runtimes = [nb_digits_runtimes, nb_faces_runtimes]
    nb_rights = [nb_digits_rights, nb_faces_rights]
    nb_wrongs = [nb_digits_wrongs, nb_faces_wrongs]
    nb_avgs = get_algo_avg(nb_runtimes, nb_rights, nb_wrongs, size)
    nb_stds = get_algo_std(nb_runtimes, nb_rights, nb_wrongs, size)

    ca_runtimes = [ca_digits_runtimes, ca_faces_runtimes]
    ca_rights = [ca_digits_rights, ca_faces_rights]
    ca_wrongs = [ca_digits_wrongs, ca_faces_wrongs]
    ca_avgs = get_algo_avg(ca_runtimes, ca_rights, ca_wrongs, size)
    ca_stds = get_algo_std(ca_runtimes, ca_rights, ca_wrongs, size)

    return [p_avgs, p_stds], [nb_avgs, nb_stds], [ca_avgs, ca_stds]

def get_algo_avg(runtimes, rights, wrongs, size):

    digits_avg= [[] for _ in range(size)]
    faces_avg= [[] for _ in range(size)]

    for i in range(size):
        digits_avg[i] = compute_avgs(runtimes[0][i], rights[0][i], wrongs[0][i])
        faces_avg[i] = compute_avgs(runtimes[1][i], rights[1][i], wrongs[1][i])

    return [digits_avg, faces_avg]


def compute_avgs(runtimes, rights, wrongs):
    runtime = round(numpy.mean(runtimes), 1)
    right = round(numpy.mean(rights), 1)
    wrong = round(numpy.mean(wrongs), 1)

    return [runtime, right, wrong]

def get_algo_std(runtimes, rights, wrongs, size):
    
    digits_std= [[] for _ in range(size)]
    faces_std= [[] for _ in range(size)]

    for i in range(size):
        digits_std[i] = compute_std(runtimes[0][i], rights[0][i], wrongs[0][i])
        faces_std[i] = compute_std(runtimes[1][i], rights[1][i], wrongs[1][i])

    return [digits_std, faces_std]

def compute_std(runtimes, rights, wrongs):
    runtime = round(numpy.std(runtimes), 1)
    right = round(numpy.std(rights), 1)
    wrong = round(numpy.std(wrongs), 1)

    return [runtime, right, wrong]

def report_statistics(p_stats, nb_stats, ca_stats, end_percent):

    print("\nReporting Averages and Standard Deviation")
    print("--------------------------Digits-------------------------    --------------------------Faces--------------------------")
    print("Train\tPerceptron\t   Naive Bayes\t   Custom Algo\t\tTrain\tPerceptron\t   Naive Bayes\t   Custom Algo")

    size = int(end_percent * 10)
    for i in range(size):

        print(str(round(0.1 * (i + 1) * 100, 2)) + "%", end="") #digits runtime
        print("\t" + str(p_stats[0][0][i][0]) + "," + str(p_stats[1][0][i][0]) + " sec\t   " + str(nb_stats[0][0][i][0]) + "," + str(nb_stats[1][0][i][0]) + " sec\t  " + str(ca_stats[0][0][i][0]) + "," + str(ca_stats[1][0][i][0]) + " sec", end="\t\t")
        print(str(round(0.1 * (i + 1) * 100, 2)) + "%", end="") #faces runtime
        print("\t" + str(p_stats[0][1][i][0]) + "," + str(p_stats[1][1][i][0]) + " sec\t   " + str(nb_stats[0][1][i][0]) +  "," + str(nb_stats[1][1][i][0]) + " sec\t  " + str(ca_stats[0][1][i][0]) + "," + str(ca_stats[1][1][i][0]) + " sec")

        #digits right
        print("\t" + str(p_stats[0][0][i][1]) + "," + str(p_stats[1][0][i][1]) + " right  " + str(nb_stats[0][0][i][1]) + "," + str(nb_stats[1][0][i][1]) + " right  " + str(ca_stats[0][0][i][1]) + "," + str(ca_stats[1][0][i][1]) +  " right", end="\t")
        #faces right
        print("\t" + str(p_stats[0][1][i][1]) + "," + str(p_stats[1][1][i][1]) + " right  " + str(nb_stats[0][1][i][1]) + "," + str(nb_stats[1][1][i][1]) + " right  " + str(ca_stats[0][1][i][1]) + "," + str(ca_stats[1][1][i][1]) +  " right")

        #digits wrong
        print("\t" + str(p_stats[0][0][i][2]) + "," + str(p_stats[1][0][i][2]) + " wrong  " + str(nb_stats[0][0][i][2]) + "," + str(nb_stats[1][0][i][2]) + " wrong  " + str(ca_stats[0][0][i][2]) + "," + str(ca_stats[1][0][i][2]) +  " wrong", end="\t")
        #faces wrong
        print("\t" + str(p_stats[0][1][i][2]) + "," + str(p_stats[1][1][i][2]) + " wrong  " + str(nb_stats[0][1][i][2]) + "," + str(nb_stats[1][1][i][2]) + " wrong  " + str(ca_stats[0][1][i][2]) + "," + str(ca_stats[1][1][i][2]) +  " wrong")

def run_and_report(algorithms, digits_dataset, faces_dataset, end_percent):

    summary = []
    percent = 0.1

    while percent <= end_percent:

        print("Training on " + str(round(percent * 100, 2)) + "% of training data")
      
        #randomly sample from the starting datasets
        trimmed_digits_dataset = random_sample(digits_dataset, percent)
        trimmed_faces_dataset = random_sample(faces_dataset, percent)

        #trimmed_digits_dataset = trim_dataset(digits_dataset, percent)
        #trimmed_faces_dataset = trim_dataset(faces_dataset, percent)

        digits_results, faces_results = run_all(algorithms, trimmed_digits_dataset, trimmed_faces_dataset)

        summary.append(report_results(digits_results, faces_results, percent))
        percent = round(percent + 0.1, 2)

    return summary
    
def run_all(algorithms, digits_dataset, faces_dataset):

    digits_results = []
    faces_results = []

    for algorithm in algorithms:
        digits_train, digits_evaluate = algorithm[0], algorithm[1]
        train_time, result = run_algo(digits_dataset, digits_train, digits_evaluate)
        digits_results.append([train_time, result])

        faces_train, faces_evaluate = algorithm[2], algorithm[3]
        train_time, result = run_algo(faces_dataset, faces_train, faces_evaluate)
        faces_results.append([train_time, result])

    return digits_results, faces_results

def run_algo(dataset, train, evaluate):
    #train_dataset = trim_dataset(dataset[0], percent)
    data = dataset[1]

    start = time.time()
    train_info = train(dataset[0])
    end = time.time()

    correct, wrong = evaluate(data, train_info)

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

        print(str(round(0.1 * (i + 1) * 100, 2)) + "%", end="") #digits runtime
        print("\t" + str(summary[i][0][0][0]) + " sec\t" + str(summary[i][0][1][0]) + " sec\t" + str(summary[i][0][2][0]) + " sec", end="\t")
        print(str(round(0.1 * (i + 1) * 100, 2)) + "%", end="") #faces runtime
        print("\t" + str(summary[i][1][0][0]) + " sec\t" + str(summary[i][1][1][0]) + " sec\t\t" + str(summary[i][1][2][0]) + " sec")

        #digits right
        print("\t" + str(summary[i][0][0][1][0]) + " right\t" + str(summary[i][0][1][1][0]) + " right\t" + str(summary[i][0][2][1][0]) + " right", end="\t")
        #faces right
        print("\t" + str(summary[i][1][0][1][0]) + " right\t" + str(summary[i][1][1][1][0]) + " right\t\t" + str(summary[i][1][2][1][0]) + " right")

        #digits wrong
        print("\t" + str(summary[i][0][0][1][1]) + " wrong\t" + str(summary[i][0][1][1][1]) + " wrong\t" + str(summary[i][0][2][1][1]) + " wrong", end="\t")
        #faces wrong
        print("\t" + str(summary[i][1][0][1][1]) + " wrong\t" + str(summary[i][1][1][1][1]) + " wrong\t\t" + str(summary[i][1][2][1][1]) + " wrong\n")

    

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

    trimmed_dataset = [[[], []], list.copy(dataset[1])]
    size = int(len(dataset[0][0]) * round(percent, 3))
    
    trimmed_dataset[0][0] = dataset[0][0][:size]
    trimmed_dataset[0][1] = dataset[0][1][:size]

    return trimmed_dataset

#randomly sample up to a perent of the dataset for training
def random_sample(dataset, percent):
  
    sampled_dataset = [[[], []], list.copy(dataset[1])]
    size = int(len(dataset[0][0]) * round(percent, 3))
    
    for a in range(size):
        random_index = random.randint(0, len(dataset[0][0]) - 1)
        sampled_dataset[0][0].append(dataset[0][0][random_index])
        sampled_dataset[0][1].append(dataset[0][1][random_index])

    return sampled_dataset

def do_nothing_train(dataset):
    return 0

def do_nothing_eval(dataset, something):
    return 0, 0

if __name__ == '__main__':
    main()