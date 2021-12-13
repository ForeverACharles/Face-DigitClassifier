from read import print_entry, print_dataset
import numpy
import math
import time

def distance(item1, item2):
    return numpy.linalg.norm(numpy.subtract(item1,item2))

def k_neighbors(dataset, curr_row, k):
    dists = []
    data = dataset[0]
    labels = dataset[1]
    for i in range(len(data)):
        dist = distance(curr_row, data[i])
        dists.append([dist, labels[i]])
    dists.sort(key=lambda input:input[0])
    return dists[:k]


def ca_digits_train(dataset):
    #return dataset
    ### FEATURE EXTRACTION ###
    #print(len(dataset[0]))
    total = []
    for i in range(10):
        total.append(numpy.zeros(784))

    for i in range(len(dataset[0])):
        total[dataset[1][i]] = numpy.add(total[dataset[1][i]], dataset[0][i])
    features = []
    counts = []
    for i in range(10):
        count = 0
        for k in range(784):
            if total[i][k] > len(dataset[0])/11.5 and not(features.__contains__(k)):
                count = count+1
                features.append(k)
        counts.append(count)
    #print(len(features))
    features.sort()
    #print(features)
    for i in range(len(dataset[0])):
        temp = []
        for j in range(len(features)):
            temp.append(dataset[0][i][features[j]])
        dataset[0][i] = temp
    



    dataset.append(features)
    return dataset

def ca_digits_evaluate(dataset, train_dataset):
    start =time.time()
    features = train_dataset[2]
    k = 5
    data = dataset[0]
    labels = dataset[1]
    successes, fails = 0, 0
    counts = [0]*10
    for i in range(len(data)):
        temp = []
        for j in range(len(features)):
            temp.append(data[i][features[j]])
    #data[i] = temp
        vals = numpy.zeros(10)
        ret = k_neighbors(train_dataset, temp, k)
        for j in range(k):
            vals[ret[j][1]] = vals[ret[j][1]] + 1
        max_count = -1
        max_val = -1
        for j in range(10):
            if vals[j] > max_count:
                max_val = j
                max_count = vals[j]
        if max_val == labels[i]:
            successes = successes + 1
        else:
            counts[labels[i]] = counts[labels[i]]+1
            fails = fails+1 
    #print(counts)
    end=time.time()
    print("==============")
    print(end-start)
    print("===============")
    return successes, fails

def ca_faces_train(dataset):
    return dataset

def ca_faces_evaluate(dataset, train_dataset):
    k = 13
    data = dataset[0]
    labels = dataset[1]
    successes, fails = 0, 0
    for i in range(len(data)):
        vals = numpy.zeros(2)
        ret = k_neighbors(train_dataset, data[i], k)
        for j in range(k):
            vals[ret[j][1]] = vals[ret[j][1]] + 1
        max_count = -1
        max_val = -1
        for j in range(2):
            if vals[j] > max_count:
                max_val = j
                max_count = vals[j]
        if max_val == labels[i]:
            successes = successes + 1
        else:
            fails = fails+1 

    return successes, fails