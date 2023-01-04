import math


training = []
no_of_cols=0   #no of different features in data
kpredicted = {}
# reading training data from text file
with open("1c_train.txt") as fic:
    for line in fic:
        line = line.rstrip(",\r\n")
        patt = line.replace('(', '').replace(')', '').replace(' ', '').strip()
        row = list(patt.split(","))
        no_of_cols=len(row)-2  #subtract 2 to remove age and class column from dataset
        training.append(row)
    for x in range(len(training)):
        for y in range(no_of_cols):
                training[x][y] = float(training[x][y])
    for j in training:
        del j[2]   # remove the age attribute from training data set
    print("Training Data:")
    print(training)
total_rows=len(training)


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
def getKNeighbors(trainingData,  k,metric):
    prediction_correct=0
    prediction_incorrect = 0
    i=0
    while(i!=len(trainingData)):
        d = []
        testData = trainingData[i] # during each iteration take each data from training set as test data point
        for j in range(len(trainingData)):
            count = []
            if(i!=j):  #check to omit the test data point from training set
                if metric == "Eucledian":
                    dist = euclideanDistance(trainingData[j], testData,no_of_cols)
                d.append([dist, j]) #appending all the distance values in d list with row number of the data
        d.sort()  #sort the distances in ascending order to find the k shortest distance
        d = d[0:k]  #select k nearest distance points from d
        neighbor_dist = d
        for d, j in d: #iterate the distances array and display the corresponding data point and distance
            count.append(trainingData[j][no_of_cols])
        mostoccured = findMostFrequentOccurringClass(count) #finding the most common label
        predicted_value = mostoccured
        if(predicted_value == testData[no_of_cols]):  #checking if predicted label is same as actual label
            prediction_correct+=1
        else:
            prediction_incorrect+=1

        print("\n\nNeighbor(s) of ", testData, "when k = ", k, " are ")
        for d, j in neighbor_dist:
            print(trainingData[j],"with distance ",d)
        print("Predicted output of testdata => ", testData, "when k = ", k, " is ",predicted_value)
        i+=1
        kpredicted[k]=prediction_correct  #store prediction count in dict
    return predicted_value


def knn_predction(training, k,metric):
    predicted_value = getKNeighbors(training, k,metric)
    return predicted_value



predicted_class = knn_predction(training,  1,"Eucledian")
predicted_class = knn_predction(training,  3,"Eucledian")
predicted_class = knn_predction(training,  5,"Eucledian")
predicted_class = knn_predction(training,  7,"Eucledian")
predicted_class = knn_predction(training,  9,"Eucledian")
predicted_class = knn_predction(training,  11,"Eucledian")

print("\n")
for k,count in kpredicted.items():
    print("No of Correct Prediction for k =",k," is: ",count)
print("Accuracy of prediction  for different k values:")
for k,count in kpredicted.items():
    acc=(count/total_rows) * 100
    print("k = ",k,"is ",round(acc,2))


