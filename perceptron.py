from read import print_entry, print_dataset
import numpy
def p_digits_train(dataset):
    #features will be each pixel. each number is a 28*28 picture then add 1 for the w0

    #extract data
    data = dataset[0]

    #extract labels 
    labels = dataset[1]
    
    # initialize weights to 0
    weights = []
    for i in range(10):
        weights.append([0]*(28*28 + 1))

    done = False
    count = 0

    while(not done and count < 25):
        changed = 0
        for i in range(len(data)):

            #val stores the values returned by each perceptron for numbers 0 -> 9
            val = [0]*10

            #start of by setting prediction to 0 and max to the value returned by 0. 
            max = numpy.dot(data[i], weights[0])
            prediction = 0

            
            # then loop through other perceptrons, if any of them have a higher val returned they are our prediction 
            for j in range(1, 10):
                ret = numpy.dot(data[i], weights[j])
                val[j] = ret
                if val[j] > max:
                    max = val[j]
                    prediction = j

            #if the prediction we made was wrong we need to update the weights. otherwise, just continue 
            if prediction != labels[i]:
                changed = changed + 1 
                weights[labels[i]] = numpy.add(weights[labels[i]], data[i])
                weights[prediction] = numpy.subtract(weights[prediction], data[i])
        count = count + 1
        if (changed/len(data) < 0.15 and  changed > 100) or changed/len(data) < 0.1 :
            print(str(changed) + " images caused the weights to be changed this round")
            done = True
        else:
            done = False
            print(str(changed) + " images caused the weights to be changed this round")
    return weights

def p_digits_evaluate(dataset, weights):
    #print("------------------RESULTS------------------")
    sucesses = 0
    fails = 0
    correct_guesses = [0]*10
    wrong_guesses = [0]*10
    
    # to check our accuracy, we loop through and calculate the value of each perceptron for each input.
    # if the perceptron with the highest value is the same as the label, it is a success, else it is a failure
    for i in range(len(dataset[0])):
        failed = False
        #val is the val for the correct perceptron
        val = numpy.dot(dataset[0][i], weights[int(dataset[1][i])])
        #val= calculate(dataset[0][i], weights, int(dataset[1][i]))

        for j in range(10):
            #ignore the perceptron for the correct value, as we are comparing the others to it, not it to itself
            if(j == int(dataset[1][i])):
                continue
            #val2 = calculate(dataset[0][i], weights, j)
            val2 = numpy.dot(dataset[0][i], weights[j])

            if val2 >= val:
                wrong_guesses[int(dataset[1][i])] = wrong_guesses[int(dataset[1][i])] +1
                failed = True
                break

        if failed:
            fails = fails + 1
        else:
            sucesses = sucesses + 1
            correct_guesses[int(dataset[1][i])] = correct_guesses[int(dataset[1][i])] + 1

    #print("successes: "+str(sucesses)+ "  failures: "+str(fails))
    #print()
    #for i in range(len(wrong_guesses)):
       # print("Guessed " + str(i) + "s correct: " + str(correct_guesses[i]) + ", wrong: " + str(wrong_guesses[i]) + " times.")

    return  sucesses, fails



def p_faces_train(dataset):
    #features will be each pixel. each number is a 35*35 picture then add 1 for the w0

    #extract data
    data = dataset[0]

    #extract labels 
    labels = dataset[1]
    
    # initialize weights to 0
    weights = [0]*(70*60 + 1)

    done = False
    count = 0

    while(not done and count < 25):
        changed = 0
        for i in range(len(data)):

            #start of by setting prediction to 0 and max to the value returned by 0. 
            val = numpy.dot(data[i], weights)
            #print("--")
            #print(i)
            #print(val)
            #print(labels[i])
            #print('--')
            if labels[i] == 1 and val < 0:
                changed = changed+1
                weights = numpy.add(weights, data[i])
            elif labels[i] == 0 and val >=0:
                changed = changed + 1
                weights = numpy.subtract(weights, data[i])

        count = count + 1
        if changed == 0:
        #if (changed/len(data) < 0.15 and  changed > 100) or changed/len(data) < 0.1 :
            print(str(changed) + " faces caused the weights to be changed this round")
            done = True
        else:
            done = False
            print(str(changed) + " faces caused the weights to be changed this round")
    return weights

def p_faces_evaluate(dataset, weights):
    #print("------------------RESULTS------------------")
    sucesses = 0
    fails = 0
    
    # to check our accuracy, we loop through and calculate the value of each perceptron for each input.
    # if the perceptron with the highest value is the same as the label, it is a success, else it is a failure
    for i in range(len(dataset[0])):
        failed = False
        #val is the val for the correct perceptron
        val = numpy.dot(dataset[0][i], weights)

        if (val >= 0 and dataset[1][i] == 0) or (val < 0 and dataset[1][i] == 1):
            fails = fails + 1
        else:
            sucesses = sucesses + 1

    return sucesses, fails