from numpy.core.fromnumeric import transpose
from read import print_entry, print_dataset
import numpy
import math

def nb_digits_train(dataset):
    
    #intialize pixel counting matrices
    digit_counts = [0 for _ in range(10)]
    pixel_freq = numpy.ones((10, 784), dtype=int)

    #look through all data and count pixel frequencies for each digit
    for i in range(len(dataset[0])):
       
        digit = dataset[1][i]
        digit_data = numpy.array(dataset[0][i])

        #for each occurence of a specific digit, increase by 1 the counts on pixels that are active
        pixel_freq[digit] = numpy.add(pixel_freq[digit], digit_data)
        digit_counts[digit] += 1

    #get probabilities for each of the 10 digits in the dataset
    digit_prob = numpy.array(list(map(lambda count : count / len(dataset[0]), digit_counts)))

    #calculate conditional probability for each pixel given the digit it appears in
    pixel_cond_prob = numpy.zeros((10, 784), dtype=float)
    #region_cond_prob = numpy.zeros((10, regions), dtype=float)
    for i in range(10):
        pixel_cond_prob[i] = numpy.array(list(map(lambda count: count / digit_counts[i], pixel_freq[i])))

    probabilities = [digit_prob, pixel_cond_prob]
    return probabilities

def nb_digits_evaluate(dataset, probabilities):
    data, labels = numpy.array(dataset[0], dtype=numpy.int64), dataset[1]
    digit_prob, pixel_cond_prob = probabilities[0], probabilities[1]

    successes, fails = 0, 0
    correct_guesses = [0 for _ in range(10)]
    wrong_guesses = [0 for _ in range(10)]
    scores = [0 for _ in range(10)]

    for i in range(len(data)):
        #calculate the score for each possible digit classification based on the data provided
        for e in range(10):
            scores[e] = calculate_score(data[i], digit_prob[e], pixel_cond_prob[e], 'digits')

        #get the max score and assign the data to a guessed digit
        guess = scores.index(max(scores))

        #compare guessed digit to the actual
        if guess == labels[i]:
            successes += 1
            correct_guesses[guess] += 1
        else:
            fails += 1
            wrong_guesses[guess] += 1

    return successes, fails

#takes product of digit probability and probability of every occurence of pixel in the data to get a score
def calculate_score(data, digit_prob, pixel_cond_prob, type):

    # 16 x 49
    train_regions = define_regions(pixel_cond_prob, type)
    data_regions = define_regions(data, type)

    regions = data_regions * train_regions
    
    # 16 x 1
    #regions = numpy.array(list(map(lambda region: numpy.flatten(region), regions)))
    #train_region_sums = numpy.array(list(map(lambda region: numpy.mean(region), train_regions)))
    #data_region_sums = numpy.array(list(map(lambda region: numpy.sum(region), data_regions)))

    #region_prob = data_region_sums * train_region_sums
    #for i in range(len(region_prob)):
        #region_value = region_prob[i]
        #region_prob[i] = math.log(region_value) if region_value != 0 else 0
    #region_prob = numpy.array(list(map(lambda value: numpy.log(value) if value != 0 else 0, region_prob)))

    #data_prob = data * pixel_cond_prob
    #for i in range(len(data_prob)):
        #value = data_prob[i]
        #data_prob[i] = math.log(value) if value != 0 else 0
    #data_prob = numpy.array(list(map(lambda value: numpy.log(value) if value != 0 else 0, data_prob)))

    for i in range(len(regions)):
        regions[i] = numpy.array(list(map(lambda value: numpy.log(value) if value != 0 else 0, regions[i])))

    regions = numpy.array(list(map(lambda region: numpy.mean(region), regions)))
    #region_prob = numpy.array(list(map(lambda region: numpy.mean(region), region_prob)))

    score = numpy.log(digit_prob) + numpy.sum(regions)
    #score = numpy.log(digit_prob) + numpy.sum(data_prob)
    return score

def define_regions(data, type):
    #divide up the pixel dataset into n by m regions
    #28x28 --> 4x4 regions each 7 by 7
    #28x28 --> 7x7 regions each 4 by 4
    #28x28 --> 2x2 regions each 14 by 14
    #70x60 --> 7x6 regions eaach 10 by 10
    #70X60 --> 14x12 regions each 5 by 5

    #create 2d vector to represent the feature regions, scaled based on approximate # of regions needed to represent the test data
    regions = []
    if type == 'digits':
        #gridview = data.reshape(28, 28)
        #divided_columns = numpy.lib.stride_tricks.as_strided(gridview, shape=(4, 28, 7), strides=(56, 224, 8))
        #reshaped = numpy.lib.stride_tricks.as_strided(divided_columns, shape=(4, 4, 7, 7), strides=(1568, 56, 224, 8))

        row = 14 #region height
        col = 14 #region length
        bytes = 8
        gridview = data.reshape(28, 28)
        #print(gridview.strides)
        divided_columns = numpy.lib.stride_tricks.as_strided(gridview, shape=(int(28/col), 28, row), strides=(col*bytes, 28*bytes, bytes))
        reshaped = numpy.lib.stride_tricks.as_strided(divided_columns, shape=(int(28/row), int(28/col), row, col), strides=(row*28*bytes, col*bytes, 28*bytes, bytes))
       
        for i in range(len(reshaped)):
            for j in range(len(reshaped[0])):
                regions.append(numpy.array(reshaped[i][j]).flatten())

    if type == 'faces':
        row = 10 #region height
        col = 10 #region length
        bytes = 8
        gridview = data.reshape(70, 60)
        #print(gridview.strides)
        divided_columns = numpy.lib.stride_tricks.as_strided(gridview, shape=(int(60/col), 70, row), strides=(col*bytes, 60*bytes, bytes))
        reshaped = numpy.lib.stride_tricks.as_strided(divided_columns, shape=(int(70/row), int(60/col), row, col), strides=(row*60*bytes, col*bytes, 60*bytes, bytes))
        
        #divided_columns = numpy.lib.stride_tricks.as_strided(gridview, shape=(6, 70, 10), strides=(80, 480, 8))
        #reshaped = numpy.lib.stride_tricks.as_strided(divided_columns, shape=(7, 6, 10, 10), strides=(4800, 80, 480, 8))
       
        for i in range(len(reshaped)):
            for j in range(len(reshaped[0])):
                regions.append(numpy.array(reshaped[i][j]).flatten())

    return numpy.array(regions)

def nb_faces_train(dataset):

    #intialize pixel counting matrices
    face_counts = [0 for _ in range(2)]
    pixel_freq = numpy.ones((2, 4200), dtype=int)

    #look through all data and count pixel frequencies for each face
    for i in range(len(dataset[0])):
       
        face = dataset[1][i]
        face_data = numpy.array(dataset[0][i])

        #for each occurence of a face, increase by 1 the counts on pixels that are active
        pixel_freq[face] = numpy.add(pixel_freq[face], face_data)
        face_counts[face] += 1

    #get probabilities for each of the image types in the dataset
    face_prob = numpy.array(list(map(lambda count : count / len(dataset[0]), face_counts)))

    #calculate conditional probability for each pixel given the face image it appears in

    pixel_cond_prob = numpy.zeros((2, 4200), dtype=float)
    
    for i in range(2):
        pixel_cond_prob[i] = numpy.array(list(map(lambda count: count / face_counts[i], pixel_freq[i])))

    probabilities = [face_prob, pixel_cond_prob]
    return probabilities

def nb_faces_evaluate(dataset, probabilities):

    data, labels = numpy.array(dataset[0], dtype=numpy.int64), dataset[1]
    face_prob, pixel_cond_prob = probabilities[0], probabilities[1]

    successes, fails = 0, 0
    correct_guesses = [0 for _ in range(2)]
    wrong_guesses = [0 for _ in range(2)]
    scores = [0 for _ in range(2)]

    for i in range(len(data)):
        #calculate the score for each possible face classification based on the data provided
        for e in range(2):
            scores[e] = calculate_score(data[i], face_prob[e], pixel_cond_prob[e], 'faces')

        #get the max score and assign the data to a guessed digit
        guess = scores.index(max(scores))

        #compare guessed digit to the actual
        if guess == labels[i]:
            successes += 1
            correct_guesses[guess] += 1
        else:
            fails += 1
            wrong_guesses[guess] += 1

    return successes, fails