import math


training = []
no_of_cols=0   #no of different features in data
# reading training data from text file
with open("1a_train.txt") as fic:
    for line in fic:
        line = line.rstrip(",\r\n")
        patt = line.replace('(', '').replace(')', '').replace(' ', '').strip()
        row = list(patt.split(","))
        no_of_cols=len(row)
        training.append(row)
    for x in range(len(training)):
        for y in range(no_of_cols-1):
            training[x][y] = float(training[x][y])
    print("Training Data:")
    print(training)

test = []
# reading test data from text file
with open("1a_test.txt") as fic:
    for line in fic:
        line = line.rstrip(",\r\n")
        patt = line.replace('(', '').replace(')', '').replace(' ', '').strip()
        row = list(patt.split(","))
        no_of_cols = len(row)
        test.append(row)
    for x in range(len(test)):
        for y in range(no_of_cols):
            test[x][y] = float(test[x][y])
    print("Test Data:")
    print(test)


# find eucledian distance
def euclideanDistance(training, test, no_of_cols):
    distance = 0
    for x in range(no_of_cols):
        distance += pow((training[x] - test[x]), 2)
    return math.sqrt(distance)

# find manhattan distance
def manhattanDistance(training, test, no_of_cols):
    distance = 0
    for x in range(no_of_cols):
        distance += abs(training[x] - test[x])
    return distance

# find minkowski distance
def minkowskiDistance(training, test, no_of_cols):
    distance = 0
    dist = 0
    p = 3
    r = 1 / float(p)
    for x in range(no_of_cols):
            dist += (pow(abs(training[x] - test[x]),p))
            distance = (pow(dist,r))
    return distance

#find most frequently occuring class in k nearest neighbors
def findMostFrequentOccurringClass(classlist):
    counter = 0
    mostfreqlabel = classlist[0]  #assuming first class is the most frequent one
    for i in classlist:
        cur_freq = classlist.count(i) #count for each class
        if (cur_freq > counter):
            counter = cur_freq
            mostfreqlabel = i

    return mostfreqlabel
#get k nearest neighbors
def getKNeighbors(trainingData, testData, k,metric):
    predicted_value = []
    for i in range(len(testData)):
        d = []
        count = []
        predicted_value=[]
        print("\n")
        for j in range(len(trainingData)):
            if metric == "Eucledian":
                dist = euclideanDistance(trainingData[j], testData[i],no_of_cols)
            elif metric == "Manhattan":
                dist = manhattanDistance(trainingData[j], testData[i], no_of_cols)
            else:
                dist = minkowskiDistance(trainingData[j], testData[i], no_of_cols)
            d.append([dist, j]) #appending all the distance values in d list with row number of the data
        d.sort()  #sort the distances in ascending order to find the k shortest distance
        d = d[0:k]  #select k nearest distance points from d
        neighbor_dist = d
        for d, j in d: #iterate the distances array and display the corresponding data point and distance
            count.append(trainingData[j][no_of_cols])
        mostoccured = findMostFrequentOccurringClass(count) #finding the most common label
        predicted_value.append(mostoccured)
        print("\nNeighbor(s) of ", testData[i], "when k = ", k, " are ")
        for d, j in neighbor_dist:
            print(trainingData[j],"with distance ",d)
        print("Predicted output of ", testData[i], "when k = ", k, " is ",predicted_value)
    return predicted_value


def knn_predction(training, test,k,metric): #calling the KNN algorithm
    predicted_value = getKNeighbors(training, test, k,metric)
    return predicted_value



#calling the knn algorithm with three distance metric with k values 1,3 and 7
print("Using Cartesian Distance")
predicted_class = knn_predction(training, test, 1,"Eucledian")
predicted_class = knn_predction(training, test, 3,"Eucledian")
predicted_class = knn_predction(training, test, 7,"Eucledian")
print("\n\n")
print("Using Manhattan Distance")
predicted_class = knn_predction(training, test, 1,"Manhattan")
predicted_class = knn_predction(training, test, 3,"Manhattan")
predicted_class = knn_predction(training, test, 7,"Manhattan")
print("\n\n")

print("Using Minkowski Distance")
predicted_class = knn_predction(training, test, 1,"Minkowski")
predicted_class = knn_predction(training, test, 3,"Minkowski")
predicted_class = knn_predction(training, test, 7,"Minkowski")

