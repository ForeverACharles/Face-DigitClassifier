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
    # KNN doesn't have "training" like most machine learning algorithms. However, it is EXTREMELY slow, therefore we instead use it to limit the number of pixels we will have to look at in the evaluation phase


    ### FEATURE EXTRACTION ###
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
    
    for i in range(len(dataset[0])):
        temp = []
        temp2 = [0]*60
        for j in range(70):
            num = 0
            for k in range(60):
                temp2[k] = temp2[k] + dataset[0][i][j*60 + k]
                num = num + dataset[0][i][j*60 + k]
            temp.append(num)
        for j in range(60):
            temp.append(temp2[j])
        dataset[0][i] = temp
    
    
    return dataset
    total = numpy.zeros(60*70)
    #for i in range(1):
    #   total.append(numpy.zeros(60*70))

    for i in range(len(dataset[0])):
        #total[dataset[1][i]] = numpy.add(total[dataset[1][i]], dataset[0][i])
        total = numpy.add(total, dataset[0][i])
    features = []
    counts = []
    for i in range(1):
        count = 0
        for k in range(60*70):
            #0.075 : 97.833 right on average
            if total[k] > len(dataset[0])*0.12 and not(features.__contains__(k)):
                count = count+1
                features.append(k)
        counts.append(count)
    features.sort()
    print(len(features))
    for i in range(len(dataset[0])):
        temp = []
        for j in range(len(features)):
            temp.append(dataset[0][i][features[j]])
        dataset[0][i] = temp
    
    dataset.append(features)
    return dataset
    

def ca_faces_evaluate(dataset, train_dataset):
    k = int(math.sqrt(len(dataset[1])))
    #features = train_dataset[2]
    if k %2 == 0:
        k = k+1
    print(k)
    data = dataset[0]
    labels = dataset[1]
    print(labels)
    successes, fails = 0, 0
    face_fails = 0
    face_successes = 0
    for i in range(len(data)):
        temp = []
        temp2 = [0]*60
        for j in range(70):
            num = 0
            for z in range(60):
                temp2[z] = temp2[z] + dataset[0][i][j*60 + z]
                num = num + data[i][j*60 + z]
            temp.append(num)
        for j in range(60):
            temp.append(temp2[j])
        #dataset[0][i] = temp
        #for j in range(len(features)):
        #    temp.append(data[i][features[j]])
        vals = numpy.zeros(2)
        ret = k_neighbors(train_dataset, temp, k)
        #print("--")
        #print(ret)
        #print("--")
        for j in range(k):
            #print(ret[j][1])
            #print(j)
            vals[ret[j][1]] = vals[ret[j][1]] + 1
        max_count = -1
        max_val = -1
        for j in range(2):
            if vals[j] > max_count:
                max_val = j
                max_count = vals[j]

       
        if max_val == labels[i]:
            if labels[i] == 1:
                face_successes = face_successes + 1
            #else:
            #    print(labels[i])
            successes = successes + 1
            
        else:
            if labels[i] == 1:
                face_fails = face_fails + 1
            #else:
            #    print(labels[i])
            fails = fails+1 

    nf_fails = fails - face_fails
    nf_suc = successes - face_successes
    print("FACES: wrong "+ str(face_fails) + " times, right " + str(face_successes) + " times")
    
    print("NOT FACES: wrong "+ str(nf_fails) + " times, right " + str(nf_suc) + " times")

    
    return successes, fails