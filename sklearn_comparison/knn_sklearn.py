from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def calculate_accuracy(y_true, y_predicted):
    good = 0
    for x in range(len(y_true)):
        if y_true[x] == y_predicted[x]:
            good += 1
    return (good / len(y_true)) * 100

# Read data
train = pd.read_csv("../data/training.txt", sep=",", header=None)
test = pd.read_csv("../data/test.txt", sep=",", header=None)

# Train/Test split
x_train, y_train = train.ix[:, :3], train.ix[:, 4]
x_test, y_test = test.ix[:, :3], test.ix[:, 4]

# Create and fit model
knn = KNeighborsClassifier(n_neighbors=3, p=2)
knn.fit(x_train, y_train)

# Predict classes of points
y_predicted = knn.predict(x_test)

print("Training samples: %s ||| Test samples: %s" % (len(x_train), len(x_test)))
print("Score: %s" % calculate_accuracy(y_test, y_predicted))