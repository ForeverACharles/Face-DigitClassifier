from read import print_entry, print_dataset

def feature_calc(data, num):
    #numbers start at 1, therefore subtract 1
    num = num -1

    # if the pixel isnt empty, return 1. otherwise return 0
    if data[int(num/28)][num%28] != ' ':  # num/28 gives us the row. num%28 gives the column
        return 1
    return 0


def calculate(image, weights):
    # sum up w0 + w1* feature1(x) + w2*feature2(x) ... etc

    #init as w0
    val = weights[0]
    features = []
    # for the rest, calculate the feature's value for the data and multiply it by the weight. add that to the current val
    for i in range(1, len(weights)):
        feature_val = feature_calc(image, i)
        val = val + (feature_val * weights[i])

        #add feature val to features for use later in the updating section
        features.append(feature_val)
    return val, features


def p_train(dataset):
    #features will be each pixel. each number is a 28*28 picture then add 1 for the w0
    
    # initialize weights to 0
    weights = [0]*(28*28+1)


    done = False
    count = 0
    #as long as we changed any of the weights when passing over them we should continue
    # however since the code doesn't seem to work this never happens, this is why count is introduced to stop infinite loops 
    while(not done and count < 7):
        changed = False
        for i in range(len(dataset[0])):

            # calculates f(xi, w) and returns value of features as well to be used in updating weights 
            val, features = calculate(dataset[0][i], weights)


            # if the value it returns is too low, we should increase the weights based on the features
            if int(dataset[1][i]) > int(round(val)) :
                changed = True  #since we had to change a weight, we will have to do another pass
                weights[0] = weights[0] + 1
                for j in range(1, len(weights)):
                    # since there is one more weight than there are features, we do features[j-1] to access the feature we want
                    weights[j] = weights[j] + features[j-1]

            
            # if the value it returns is too high, we should decrease the weights based on the features
            elif int(dataset[1][i]) < int(round(val)) :
                changed = True #since we had to change a weight, we will have to do another pass
                weights[0] = weights[0] - 1
                for j in range(1, len(weights)):
                    # since there is one more weight than there are features, we do features[j-1] to access the feature we want
                    weights[j] = weights[j] - features[j-1]
        done = not changed
        count = count + 1
    return weights

def p_evaluate(dataset, weights):
    # in order to test the weights we found, we simply calculate the value using the weights and then compare it to the labels. if they are equal, we succeeded. if not we failed
    sucesses = 0
    fails = 0
    for i in range(len(dataset[0])):
        val, features = calculate(dataset[0][i], weights)
        if int(dataset[1][i]) == int(round(val)):
            #print("val: "+ str(val) + "   label: "+ str(dataset[1][i]))
            sucesses = sucesses + 1
        else:
            fails = fails + 1
            #print("val: "+ str(val) + "   label: "+ str(dataset[1][i]))
    print("successes: "+str(sucesses)+ "  failures: "+str(fails))

    return 0