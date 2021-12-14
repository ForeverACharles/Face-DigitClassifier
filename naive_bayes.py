from read import print_entry, print_dataset
import numpy

def nb_digits_train(dataset):
    
    #intialize pixel counting matrices
    digit_counts = [0 for _ in range(10)]
    pixel_freq = numpy.zeros((10, 784), dtype=int)

    #look through all data and count pixel frequencies for each digit
    for i in range(len(dataset[0])):
       
        digit = dataset[1][i]
        digit_data = numpy.array(dataset[0][i])

        #for each occurence of a specific digit, increase by 1 the counts on pixels that are active
        pixel_freq[digit] = numpy.add(pixel_freq[digit], digit_data)
        digit_counts[digit] += 1

    #get probabilities for each of the 10 digits in the dataset
    digit_prob = numpy.array(list(map(lambda count : count / len(dataset[0]), digit_counts)))
    #print(digit_prob)

    #calculate conditional probability for each pixel given the digit it appears in
    regions = 784

    pixel_cond_prob = numpy.zeros((10, 784), dtype=float)
    #region_cond_prob = numpy.zeros((10, regions), dtype=float)
    for i in range(10):
        pixel_cond_prob[i] = numpy.array(list(map(lambda count: count / digit_counts[i], pixel_freq[i])))
        #region_cond_prob[i] = split_regions(pixel_cond_prob[i], regions)
        #pixel_cond_prob[i] = upscale_regions(region_cond_prob[i], 784)

    #print(pixel_cond_prob[0])    

    probabilities = [digit_prob, pixel_cond_prob]
    return probabilities

def nb_digits_evaluate(dataset, probabilities):
    data, labels = dataset[0], dataset[1]
    digit_prob, pixel_cond_prob = probabilities[0], probabilities[1]

    successes, fails = 0, 0
    correct_guesses = [0 for _ in range(10)]
    wrong_guesses = [0 for _ in range(10)]
    scores = [0 for _ in range(10)]

    for i in range(len(data)):
        #calculate the score for each possible digit classification based on the data provided
        for e in range(10):
            scores[e] = calculate_score(data[i], digit_prob[e], pixel_cond_prob[e])

        #get the max score and assign the data to a guessed digit
        guess = scores.index(max(scores))

        #compare guessed digit to the actual
        if guess == labels[i]:
            successes += 1
            correct_guesses[guess] += 1
        else:
            fails += 1
            wrong_guesses[guess] += 1

        #print("successes: "+str(successes)+ "  failures: "+str(fails))
        #print()
        #for i in range(len(wrong_guesses)):
            #print("Guessed " + str(i) + "s correct: " + str(correct_guesses[i]) + ", wrong: " + str(wrong_guesses[i]) + " times.")
    return successes, fails

#takes product of digit probability and probability of every occurence of pixel in the data to get a score
def calculate_score(data, digit_prob, pixel_cond_prob):
    data_prob = data * pixel_cond_prob
    #data_regions = split_regions(data, 392)
    #data_prob = data_regions * pixel_cond_prob

    data_prob = numpy.where(data_prob == 0, 1, data_prob) #replace probabities of 0 with 1, so that score is unaffected by pixels that are not active
    score = digit_prob * numpy.prod(data_prob)
    
    return score

def split_regions(data, num_regions):
    region_size = int(len(data) / num_regions)

    regions = numpy.zeros(num_regions, dtype=float)

    for r in range(len(regions)):
        index = r * int(region_size)
        regions[r] = numpy.sum(data[index:index + region_size]) / region_size

    return regions

def upscale_regions(regions, upscale_size):
    upsacled = numpy.zeros(upscale_size, dtype=float)
    region_size = int(upscale_size / len(regions))

    for r in range(len(regions)):
        index = r * region_size
        upsacled[index: index + region_size] = regions[r]

    return upsacled

def nb_faces_train(dataset):
    something = [0] * 1000
    return something

def nb_faces_evaluate(dataset, something):
    successes, fails = 0, 0
    return successes, fails