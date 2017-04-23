from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def main():
    knn = KNeighborsClassifier(n_neighbors=3, p=2)
    path_to_training_data="../data/training.txt"
    path_to_test_data="../data/test.txt"

    train = pd.read_csv(path_to_training_data, sep=",", header = None)
    test = pd.read_csv(path_to_test_data, sep=",", header = None)

    x_train = train.ix[:,:3]
    x_test = test.ix[:,:3]
    y_train = train.ix[:,4]
    y_test = test.ix[:,4]

    """
        Fit the model using X as training data and y as target values
    """
    knn.fit(x_train, y_train)
    """
    	Predict the class labels for the provided data
    """
    y_predicted = knn.predict(x_test)

    good = 0
    for x in range(len(y_test)):
        if y_test[x] == y_predicted[x]:
            good += 1
    print("score:" + str(good / len(y_test)*100)+"%")

if __name__ == "__main__":
    main()