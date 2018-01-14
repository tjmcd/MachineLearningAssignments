""""
Week 01 Prove
Author: Ted McDaniel
"""

# HardCodedModel: predict always returns a 0
class HardCodedModel:
    def predict(self, the_test_data):
        target_predictions = []
        for data in the_test_data:
            target_predictions.append(0)
        return target_predictions

# HardCodedClassifier: Nothing intelligent (yet)
class HardCodedClassifier:
    def fit(self, the_train_data, the_train_targets):
        return HardCodedModel()

# Thanks for the help Bro. Burton
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Step 1: Load data
iris = datasets.load_iris()

# Step 2: Split data into train/test sets
train_data, test_data, train_targets, test_targets = train_test_split(iris.data, iris.target, test_size=0.3)

# # Step 3: Use existing algorithm
# classifier = GaussianNB()
# model = classifier.fit(train_data, train_targets)
#
# # Step 4: Use model for predictions
# predicted_targets = model.predict(test_data)

# Step 5: Use HardCodedClassifier and HardCodedModel
classifier = HardCodedClassifier()
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
