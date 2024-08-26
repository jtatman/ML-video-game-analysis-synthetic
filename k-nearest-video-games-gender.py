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
    This version takes a different tack - we predict the users gender given the example of type of video games they play, turining the previous script on an axis. Cross referencing like this shows the applicability of the same modeling to different aspects of interest in the completed system. As before, the data can be used to suggest, or suggest marketing based on a common pool of interest.
    This script shows the possibility of expanding this into clustering, which shows different clusters of commonality amongst interesting data. In this case, we would reduce the dimensionality to a flat dimensional structure, and then graph the results to visualize the clustering of data in a dataset. 
'''

import pandas as pd
import numpy as np
import random

# load data
data = pd.read_csv('video-game-dataset.csv')
# debug
# print(data.columns)
data = data[['age', 'height', 'weight', 'style', 'gender']] 
# debug 
# print(data.columns)

class_map = {'Strategy': 0, 'Platformer': 1, 'Action': 2, 'RPG': 3}
rev_class_map = {0: 'Strategy', 1: 'Platformer', 2: 'Action', 3: 'RPG'}
# we'll use this later
gender_map = {0: 'male', 1: 'female'}

data['style'] = data['style'].map(class_map)

#data['gender'] = data['gender'].map(gender_mapping)

# X, y where X is data and y is labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# initialize value of k
# k gives {x} number of results 
k = 38

# calculate Euclidean distance
# euclidean distance is the straightest distance between 
# two points in euclidean space of simple dimensions
# this should be contrasted with other distance calculations 
# where input is vectors
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
    # debug
    # print(f"neighbors is: {neighbors}")
    classes = [y[i] for i in neighbors]
    # debug
    # print(f"classes is: {classes}")
    # get most frequent by using max values of top classes count
    most_frequent = max(set(classes), key=classes.count)
    # debug
    # print(f"frequent is: {most_frequent}")
    return most_frequent

def gen_test_data():
    # set some reasonable ranges
    # a = age, h = height, w = weight, s = style
    min_a = 5 
    max_a = 40
    min_h = 48
    max_h = 78
    min_w = 40
    max_w = 250
    min_s = 0
    max_s = 3
    a = random.randint(min_a, max_a)
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    s = random.randint(min_s, max_s)
    return np.array([ a, h, w, s ])

# test implementation with a sample data point
print(f"columns are: age, height, weight, style of play")
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


print(f"Given age: {test_data[0]}\nGiven height: {test_data[1]}\nGiven weight: {test_data[2]}\nGiven style: {rev_class_map[test_data[3]]}\nPredicted gender: {gender_map[int(predict(test_data, k))]}")
print(f"Given age: {test_data2[0]}\nGiven height: {test_data2[1]}\nGiven weight: {test_data2[2]}\nGiven style: {rev_class_map[test_data2[3]]}\nPredicted gender: {gender_map[int(predict(test_data2, k))]}")
print(f"Given age: {test_data3[0]}\nGiven height: {test_data3[1]}\nGiven weight: {test_data3[2]}\nGiven style: {rev_class_map[test_data3[3]]}\nPredicted gender: {gender_map[int(predict(test_data3, k))]}")
print(f"Given age: {test_data4[0]}\nGiven height: {test_data4[1]}\nGiven weight: {test_data4[2]}\nGiven style: {rev_class_map[test_data4[3]]}\nPredicted gender: {gender_map[int(predict(test_data4, k))]}")
print(f"Given age: {test_data5[0]}\nGiven height: {test_data5[1]}\nGiven weight: {test_data5[2]}\nGiven style: {rev_class_map[test_data5[3]]}\nPredicted gender: {gender_map[int(predict(test_data5, k))]}")


