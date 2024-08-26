''' k-nearest neighbors from scratch

steps: 

- Load the data;
- Initialise the value of k;
- For getting the predicted class, iterate from 1 to total number of training data points;
- Calculate the distance between test data and each row of training data;
- Sort the calculated distances in ascending order based on distance values;
- Get top k rows from the sorted array;
- Get the most frequent class of these rows; and
- Return the predicted class.

synopsis:
    This mimics loosely the training of models for predicitve ability. We use a dataset for ranked predictions and then predict based on the similarity and frequency of similar results, or the overall distribution in other words.
    This version predicts the style of gaming the user would use given the predictive data. 
    These exercises are very important, because they open the gateway to recommendation systems. By using data to predict user preferences, we can target the user with suggestions based on seemingly unrelated data patterns. One important but banal aspect for recommendation systems is marketing. Another, is a suggestions system on par with Spotify, although Spotify uses specific and proprietary algorithms that are quite performant. 

'''

import pandas as pd
import numpy as np
import random

# load data
data = pd.read_csv('video-game-dataset.csv')
print(data.describe())

# X, y where X is data and y is labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# initialize value of k
# k gives {x} number of results 
k = 38

# calculate Euclidean distance
# euclidean distance is the straightest distance between 
# two points in euclidean space of two or more dimensions
# this should be contrasted with other distance calculations 
# where input is vectors, i.e. overly simplified, points plus direction 
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# get k-nearest neighbors
def get_neighbors(test_data, k):
    distances = []
    # iterate from one to total number of data points (length of range)
    for i in range(len(X)):
        dist = euclidean_distance(test_data, X[i])
        distances.append((i, dist))
    # sort distances based on k-nn value
    distances.sort(key=lambda x: x[1])
    # get k top results (after sorting)
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# predict class of a test data point
def predict(test_data, k):
    neighbors = get_neighbors(test_data, k)
    classes = [y[i] for i in neighbors]
    # get most frequent by using max values of top classes count
    most_frequent = max(set(classes), key=classes.count)
    return most_frequent

def gen_test_data():
    # set some reasonable ranges
    # a = age, h = height, w = weight
    min_a = 5 
    max_a = 40
    min_h = 48
    max_h = 78
    min_w = 40
    max_w = 250
    gen_m = 0
    gen_f = 1
    a = random.randint(min_a, max_a)
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    g = random.randint(gen_m, gen_f)
    return np.array([ a, h, w, g ])

# test implementation with a sample data point
print(f"data columns: age, height, weight, gender")
test_data = gen_test_data()
print(f"test data 1: {test_data}")
test_data2 = gen_test_data()
print(f"test data 2: {test_data2}")
test_data3 = gen_test_data()
print(f"test data 3: {test_data3}")
test_data4 = gen_test_data()
print(f"test data 4: {test_data4}")
test_data5 = gen_test_data()
print(f"test data 5: {test_data5}")


print(f"Given age: {test_data[0]}\nGiven height: {test_data[1]}\nGiven weight: {test_data[2]}\nGiven gender: {'male' if test_data[3] == 0 else 'female'}\nPredicted class: {predict(test_data, k)}")
print(f"Given age: {test_data2[0]}\nGiven height: {test_data2[1]}\nGiven weight: {test_data2[2]}\nGiven gender: {'male' if test_data2[3] == 0 else 'female'}\nPredicted class: {predict(test_data2, k)}")
print(f"Given age: {test_data3[0]}\nGiven height: {test_data3[1]}\nGiven weight: {test_data3[2]}\nGiven gender: {'male' if test_data3[3] == 0 else 'female'}\nPredicted class: {predict(test_data3, k)}")
print(f"Given age: {test_data4[0]}\nGiven height: {test_data4[1]}\nGiven weight: {test_data4[2]}\nGiven gender: {'male' if test_data4[3] == 0 else 'female'}\nPredicted class: {predict(test_data4, k)}")
print(f"Given age: {test_data5[0]}\nGiven height: {test_data5[1]}\nGiven weight: {test_data5[2]}\nGiven gender: {'male' if test_data5[3] == 0 else 'female'}\nPredicted class: {predict(test_data5, k)}")


