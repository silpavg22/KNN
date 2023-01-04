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
    d = []
    count = []
    predicted_value=[]
    print("\n")
    for j in range(len(trainingData)):
        if metric == "Eucledian":
            dist = euclideanDistance(trainingData[j], testData,no_of_cols)
        d.append([dist, j]) #appending all the distance values in d list with row number of the data
    d.sort()   #sort the distances in ascending order to find the k shortest distance
    d = d[0:k]  #select k nearest distance points from d
    neighbor_dist = d
    for d, j in d: #iterate the distances array and display the corresponding data point and distance
        count.append(trainingData[j][no_of_cols])
    mostoccured = findMostFrequentOccurringClass(count)  # finding the most common label
    predicted_value.append(mostoccured)
    print("\n\nNeighbor(s) of ", testData, "when k = ", k, " are ")
    for d, j in neighbor_dist:
        print(trainingData[j],"with distance ",d)
    print("Predicted output of ", testData, "when k = ", k, " is ",predicted_value)
    return predicted_value


def knn_prediction(training, test,k,metric): #calling the KNN algorithm
    predicted_value = getKNeighbors(training, test, k,"Eucledian")
    return predicted_value


test=[]  #taking test data input from user
for i in range(no_of_cols):
    print("Enter test data value ",i+1," :")
    num=input ()
    test.append(float(num))

k=int(input("Enter value for k:")) #taking k value as input from user

predicted_class = knn_prediction(training, test, k,"Eucledian")
print("\nPredicted Class for k = ", k, 'is', predicted_class)

