import csv
import math
import operator


class KNearestNeighbors:
    """
    Main class for points classification
    """
    def __init__(self, path_to_training_data="../data/training.txt", path_to_test_data="../data/test.txt", k_value=3):
        """
        :param KNearestNeighbors self: class instance
        :param str path_to_training_data: path to file
        :param str path_to_test_data: path to file
        :param int k_value: Number of nearest neighbors
        """
        self.list_of_training_points = []
        self.list_of_test_points = []
        self.k = k_value
        self.load_data_from_csv(path_to_training_data, training=True)
        self.load_data_from_csv(path_to_test_data, training=False)

    def load_data_from_csv(self, path, training):
        """
        Method for load tada from csv files
        :param KNearestNeighbors self: class instance 
        :param string path: path to file
        :param bool training: traning is enabled or not
        """
        with open(path, mode="rt", encoding="utf-8") as file:  # Open file with text mode
            rows = csv.reader(file)
            for row in rows:
                point_values = []
                if len(row) == 0:  # Don't create the Point instance if the row is empty
                    continue
                for number_of_value in range(len(row)-1):  # extract values
                    point_values.append(row[number_of_value])
                point_class = row[len(row)-1]
                if training is True:
                    # Create the point instance and append the list of points
                    self.list_of_training_points.append(Point(point_values, point_class))
                else:
                    self.list_of_test_points.append(Point(point_values, point_class))

    @staticmethod
    def calculate_euclidean_distance(point_1, point_2):
        """
        method for calculate eucliedan distance
        :param KNearestNeighbors self: class instance
        :param  Point point_1: first point object
        :param  Point point_2: second point object
        :return: distance
        :rtype: float
        """
        distance = 0.0
        dimensions = len(point_1.list_of_values)
        for i in range(dimensions):
            distance += pow((point_1.get_point_value(i) - point_2.get_point_value(i)), 2)
        return math.sqrt(distance)

    def get_k_nearest_neighbors(self, number_of_tested_point):
        """
        Method which is finding K nearest points for current point 
        :param KNearestNeighbors self: class instance
        :param  int number_of_tested_point: inex of point
        :return: k_nearest_neighbors 
        :rtype: list
        """
        training_points_and_distances = []
        tested_point = self.list_of_test_points[int(number_of_tested_point)]
        for training_point in self.list_of_training_points:
            distance = self.calculate_euclidean_distance(tested_point, training_point)  # calculate every distance
            training_points_and_distances.append((training_point, distance))
        training_points_and_distances.sort(key=operator.itemgetter(1))  # Sorting by the lowest distance
        k_nearest_neighbors = []
        for i in range(self.k):
            k_nearest_neighbors.append(training_points_and_distances[i][0])  # extract points with the lowest distance
        return k_nearest_neighbors

    @staticmethod
    def get_prediction(k_nearest_neighbors):  # predicts class basing on classes of k nearest neighbours
        """
        Method for point classification
        :param list k_nearest_neighbors: K nearest points
        :return: str prediction
        :rtype: list
        """
        scores = {"Iris-virginica": 0,
                  "Iris-versicolor": 0,
                  "Iris-setosa": 0}
        for point in k_nearest_neighbors:
            point_class = point.get_point_class()
            scores[point_class] += 1  # gives score if point has certain class
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        prediction = sorted_scores[0][0]
        return prediction

    @staticmethod
    def calculate_accuracy(y_true, y_predicted):  # Prediction classes will be taken from 'pointInstance.get_point_class()'
        """
        Class for calculate accuracy of classification
        :param list y_true: list of true classes of points
        :param list y_predicted: list of predicted classed of points
        :return: str prediction
        :rtype: float
        """
        good = 0
        for x in range(len(y_true)):
            if y_true[x] == y_predicted[x]:
                good += 1
        return (good / len(y_true)) * 100  # calculate percent of well predicted points


class Point:
    """
    Point object model
    """
    def __init__(self, list_of_values, point_class):
        """
        object initialization
        :param Point self: 
        :param list list_of_values: list of point values
        :param str point_class: point class
        """
        self.list_of_values = list_of_values
        self.point_class = point_class

    def __str__(self):
        """
        method for returning string about object data
        :param Point self: 
        :param str point_class:
        :return: string
        :rtype: string
        """
        return "Values: " + str(self.get_point_values()) + ", Class: " + str(self.get_point_class())

    def get_point_value(self, number_of_value):
        """
        method for getting point value by index
        :param Point self: class instance
        :param int number_of_value: index of value
        :return: exception or float
        """
        if number_of_value > len(self.list_of_values):
            raise Exception("Too big number of value")
        elif number_of_value < 0:
            raise Exception("Too small number of value")
        else:
            return float(self.list_of_values[number_of_value])

    def get_point_values(self):
        """
        getter
        :param Point self: class instance
        :return: list_of_values: values list
        :rtype list:
        """
        return self.list_of_values

    def get_point_class(self):
        """
        getter
        :param Point self:  class instance
        :return: point_class: class of point
        :rtype string:
        """
        return self.point_class


def main():
    """
    Main method of application
    """
    knn = KNearestNeighbors()

    y_predicted = []
    for i in range(len(knn.list_of_test_points)):  # predict class for every point in test data
        neighbour = knn.get_k_nearest_neighbors(i)
        y_predicted.append(knn.get_prediction(neighbour))

    y_true = []
    for point in knn.list_of_test_points:  # get real class for every point in test data
        y = point.get_point_class()
        y_true.append(y)

    training_points = len(knn.list_of_training_points)
    test_points = len(knn.list_of_test_points)

    print("Training samples: %s ||| Test samples: %s" % (training_points, test_points))
    print("Score: %s" % knn.calculate_accuracy(y_true, y_predicted))

if __name__ == "__main__":
    main()
