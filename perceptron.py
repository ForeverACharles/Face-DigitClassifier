from read import print_entry, print_dataset

def feature_calc(data, num):
    #numbers start at 1, therefore subtract 1
    num = num -1

    # if the pixel isnt empty, return 1. otherwise return 0
    if data[int(num/28)][num%28] != ' ':  # num/28 gives us the row. num%28 gives the column
        return 1
    return 0


def calculate(image, weights, label):
    # sum up w0 + w1* feature1(x) + w2*feature2(x) ... etc

    #init as w0
    val = weights[label][0]
    features = []
    # for the rest, calculate the feature's value for the data and multiply it by the weight. add that to the current val
    for i in range(1, len(weights[label])):
        feature_val = feature_calc(image, i)
        val = val + (feature_val * weights[label][i])

        #add feature val to features for use later in the updating section
        features.append(feature_val)
    return val, features


def p_train(dataset):
    #features will be each pixel. each number is a 28*28 picture then add 1 for the w0
    
    # initialize weights to 0
    weights = []
    for i in range(10):
        weights.append([0]*(28*28+1))


    done = False
    count = 0
    #as long as we changed any of the weights when passing over them we should continue
    # however since the code doesn't seem to work this never happens, this is why count is introduced to stop infinite loops 
    while(not done and count < 5):
        changed = False
        for i in range(len(dataset[0])):

            # calculates f(xi, w) and returns value of features as well to be used in updating weights 
            


            # if the value it returns is too low, we should increase the weights based on the features
            for j in range(10):
                val, features = calculate(dataset[0][i], weights, j)
                if j == int(dataset[1][i]) and val < 0 :
                    changed = True  #since we had to change a weight, we will have to do another pass
                    weights[int(dataset[1][i])][0] = weights[int(dataset[1][i])][0] + 1
                    for k in range(1, len(weights[int(dataset[1][i])])):
                        # since there is one more weight than there are features, we do features[j-1] to access the feature we want
                        weights[int(dataset[1][i])][k] = weights[int(dataset[1][i])][k] + features[k-1]


                # if the value it returns is too high, we should decrease the weights based on the features
                elif j != int(dataset[1][i]) and val >= 0 :
                    changed = True #since we had to change a weight, we will have to do another pass
                    weights[j][0] = weights[j][0] - 1
                    for k in range(1, len(weights[int(dataset[1][i])])):
                        # since there is one more weight than there are features, we do features[j-1] to access the feature we want
                        weights[j][k] = weights[j][k] - features[k-1]
        done = not changed
        count = count + 1
    return weights

def p_evaluate(dataset, weights):
    # in order to test the weights we found, we simply calculate the value using the weights and then compare it to the labels. if they are equal, we succeeded. if not we failed
    sucesses = 0
    fails = 0
    for i in range(len(dataset[0])):
        failed = False
        #print('----------')
        val, features = calculate(dataset[0][i], weights, int(dataset[1][i]))
        #print("actual("+dataset[1][i]+"): "+ str(val))
        for j in range(10):
            if(j == int(dataset[1][i])):
                continue
            val2, features = calculate(dataset[0][i], weights, j)
            if(int(dataset[1][i])==8):
                print(str(j) + "   val: "+ str(val2))
            if val2 > val:
                failed = True
                break
        if failed:
            fails = fails + 1
        else:
            print("actual("+dataset[1][i]+"): "+ str(val))
            sucesses = sucesses + 1
        #print("------------")
    print("successes: "+str(sucesses)+ "  failures: "+str(fails))

    return 0