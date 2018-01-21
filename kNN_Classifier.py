""""
Week 02 Prove
Author: Ted McDaniel

The assignment this week is to implement a kNN classifier. So, here's mine.
"""
# Thanks for the help Bro. Burton
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
from scipy.spatial import distance

# kNNModel
class kNNModel:
    def __init__(self, the_training_data, the_training_targets):
        self.training_targets = the_training_targets
        self.training_data = the_training_data

    def predict(self, the_test_data):
        # Calculate distances
        target_predictions = []
        distances = []
        distance_dict = {}
        # test_index = iter(the_test_data)
        # train_index = 0
        for test_index, test_data_element in enumerate(the_test_data):
            for train_index, train_data_element in enumerate(self.training_data):
                dist = distance.euclidean(test_data_element, train_data_element)
                # print(dist)
                distance_dict[str(train_index)] = str(dist)
                # print(distance_dict)
                # print(distances[train_index])
                distances.append(distance_dict)
                # print(distances[test_index])
                distance_dict.clear()

        # for row in distances:
        # print(distances)

        # Sort dictionaries in list
        i = 0
        # for dict in distances:
            # distances[i] = sorted(distances[i], key=lambda k: k['key'])
        # print(distances)
        return target_predictions

# kNN Classifier
class kNNClassifier:
    def __init__(self, k_neighbors):
        self.k = k_neighbors

    def fit(self, the_train_data, the_train_targets):
        return kNNModel(the_train_data, the_train_targets)

# Load iris data and split data into train/test sets
iris = datasets.load_iris()
# These are all lists. It's hard for me to keep track of data types in Python
train_data, test_data, train_targets, test_targets = train_test_split(iris.data, iris.target, test_size=0.3)

# Step 5: Use HardCodedClassifier and HardCodedModel
classifier = kNNClassifier(k_neighbors=2)
model = classifier.fit(train_data, train_targets)
predicted_targets = model.predict(test_data)

index = 0
number_of_matches = 0
for target in predicted_targets:
    if predicted_targets[index] == test_targets[index]:
        number_of_matches += 1
    index +=1

print("Number of matches: {}".format(number_of_matches))
print("Accuracy: {:.2f}%".format( (number_of_matches/test_targets.shape[0])*100 ))

# name = input("What is your name? ")
# print(name)
