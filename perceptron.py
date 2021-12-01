from read import print_entry, print_dataset
import numpy
def calculate(image, weights, label):
    #image is already processed

    #init sum as w0
    val = weights[label][0]
    for i in range(1, len(weights[label])):
        val = val + (image[int((i-1)/28)][(i-1)%28] * weights[label][i])
    return val


    # sum up w0 + w1* feature1(x) + w2*feature2(x) ... etc

    #init sum as w0
    #val = weights[label][0]
    
    # next, calculate each feature's value for the data and multiply it by the weight. add that to the current val
    #features = []
    #for i in range(1, len(weights[label])):
    #    features.append(1 if image[int((i-1)/28)][(i-1)%28] != ' ' else 0)
    #    val = val + (features[i-1] * weights[label][i])
    #return val, features

def p_train(dataset):
    #features will be each pixel. each number is a 28*28 picture then add 1 for the w0


    #extract data
    data = dataset[0]

    #extract labels and turn them into ints 
    labels = dataset[1]
    for i in range(len(labels)):
        labels[i] = int(labels[i])
    
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

            #start of by setting prediction to 0 and max to the value returned by 0. then loop through other perceptrons, if any of them have a higher val returned they are our prediction 
            #max = calculate(data[i], weights, 0)

            max = numpy.dot(data[i], weights[0])
            prediction = 0
            for j in range(1, 10):
                #ret = calculate(data[i], weights, j)
                ret = numpy.dot(data[i], weights[j])
                val[j] = ret
                #print(val[j])
                if val[j] > max:
                    max = val[j]
                    prediction = j

            #if the prediction we made was wrong we need to update the weights. otherwise, just continue 
            if prediction != labels[i]:
                changed = changed + 1  #since we had to change a weight, we will have to do another pass


                #update the perceptron for what we should have predicted it to be (increase)
                #weights[labels[i]][0] = weights[labels[i]][0] + 1
                weights[labels[i]] = numpy.add(weights[labels[i]], data[i])
                weights[prediction] = numpy.subtract(weights[prediction], data[i])
                #for k in range(1, len(weights[labels[i]])):

                    # since there is one more weight than there are features, we do features[j-1] to access the feature we want
                #    weights[labels[i]][k] = weights[labels[i]][k] + data[i][int((k-1)/28)][(k-1)%28]


                #update the perceptron for what we actually predicted it to be (decrease)
                #weights[prediction][0] = weights[prediction][0]-1
                #for k in range(1, len(weights[prediction])):
                    
                    # since there is one more weight than there are features, we do features[j-1] to access the feature we want
                #   weights[prediction][k] = weights[prediction][k] - data[i][int((k-1)/28)][(k-1)%28]
        count = count + 1
        if (changed/len(data) < 0.15 and  changed > 100) or changed/len(data) < 0.1 :
            print(str(changed) + " images caused the weights to be changed this round")
            done = True
        else:
            done = False
            print(str(changed) + " images caused the weights to be changed this round")
    return weights





    #as long as we changed any of the weights when passing over them we should continue
    # however since the code doesn't seem to work this never happens, this is why count is introduced to stop infinite loops 
    #while(not done and count < 25):
    #    changed = 0
    #    for i in range(len(data)):
#
    #        #val stores the values returned by each perceptron for numbers 0 -> 9
    #        val = [0]*10
#
    #        #start of by setting prediction to 0 and max to the value returned by 0. then loop through other perceptrons, if any of them have a higher val returned they are our prediction 
    #        max, features = calculate(data[i], weights, 0)
    #        prediction = 0
    #        for j in range(1, 10):
    #            ret, features_temp = calculate(data[i], weights, j)
    #            val[j] = ret
    #            #print(val[j])
    #            if val[j] > max:
    #                max = val[j]
    #                prediction = j
#
    #        #if the prediction we made was wrong we need to update the weights. otherwise, just continue 
    #        if prediction != labels[i]:
    #            changed = changed + 1  #since we had to change a weight, we will have to do another pass
#
#
    #            #update the perceptron for what we should have predicted it to be (increase)
    #            weights[labels[i]][0] = weights[labels[i]][0] + 1
    #            for k in range(1, len(weights[labels[i]])):
#
    #                # since there is one more weight than there are features, we do features[j-1] to access the feature we want
    #                weights[labels[i]][k] = weights[labels[i]][k] + features[k-1]
#
#
    #            #update the perceptron for what we actually predicted it to be (decrease)
    #            weights[prediction][0] = weights[prediction][0]-1
    #            for k in range(1, len(weights[prediction])):
    #                
    #                # since there is one more weight than there are features, we do features[j-1] to access the feature we want
    #                weights[prediction][k] = weights[prediction][k] - features[k-1]
    #    count = count + 1
    #    if changed/len(data) < 0.1:
    #        print(str(changed) + " images caused the weights to be changed this round")
    #        done = True
    #    else:
    #        done = False
    #        print(str(changed) + " images caused the weights to be changed this round")
    #return weights

def p_evaluate(dataset, weights):
    print("------------------RESULTS------------------")
    sucesses = 0
    fails = 0
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

    print("successes: "+str(sucesses)+ "  failures: "+str(fails))
    print()
    for i in range(len(wrong_guesses)):
        print("Guessed " + str(i) + "'s wrong "+ str(wrong_guesses[i]) + " times.")

    return 0