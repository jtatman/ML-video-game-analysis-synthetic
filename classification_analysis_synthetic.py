import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

'''
a classifier model is appropriate for categorical labels, so let's see if the regression model performs badly when constrasted with a true classifier model

with this modeling, we see the accuracy increase by a factor of four when including synthetic data.

we see that the CTGAN model has a better performance when creating synthetic data
'''

initial_file = "video-game-dataset.csv"
synthetic_one = "video-game-dataset-extended.csv"
synthetic_two = "video-game-dataset-extended-ctgan.csv"

df_initial = pd.read_csv(initial_file)
df_syn_one = pd.read_csv(synthetic_one)
df_syn_two = pd.read_csv(synthetic_two)

# ground truth dataset
X_ground_truth = df_initial.iloc[:, :-1]
y_ground_truth = df_initial.iloc[:, -1]

# synthetic datasets
X_synthetic_1 = df_syn_one.iloc[:, :-1]
y_synthetic_1 = df_syn_one.iloc[:, -1]

X_synthetic_2 = df_syn_two.iloc[:, :-1]
y_synthetic_2 = df_syn_two.iloc[:, -1]


# first ground truth
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_ground_truth, y_ground_truth, test_size=0.2, random_state=42)

# initialize and train the RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# predict on the test set
y_pred = classifier.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"accuracy of ground truth: {accuracy}")
print("classification report for ground truth:")
print(report)

# first synthetic example
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_synthetic_1, y_synthetic_1, test_size=0.2, random_state=42)

# initialize and train the RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# predict on the test set
y_pred = classifier.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"accuracy of synthetic one: {accuracy}")
print("classification report for synthetic one:")
print(report)

# second synthetic example
# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_synthetic_2, y_synthetic_2, test_size=0.2, random_state=42)

# initialize and train the RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# predict on the test set
y_pred = classifier.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"accuracy of synthetic two: {accuracy}")
print("classification report for synthetic two:")
print(report)
