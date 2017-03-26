import csv
import math
import operator


class KNearestNeighbors:

    def __init__(self, path_to_training_data="data/training.txt", path_to_test_data="data/test.txt", k_value=3):
        self.list_of_training_points = []
        self.list_of_test_points = []
        self.k = k_value
        self.load_data_from_csv(path_to_training_data, training=True)
        self.load_data_from_csv(path_to_test_data, training=False)

    def load_data_from_csv(self, path, training):
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
        distance = 0.0
        dimensions = len(point_1.list_of_values)
        for i in range(dimensions):
            distance += pow((point_1.get_point_value(i) - point_2.get_point_value(i)), 2)
        return math.sqrt(distance)

    def get_k_nearest_neighbors(self, number_of_tested_point):
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

    def get_response(self, k_nearest_neighbors):
        # TODO
        pass

    def calculate_accuracy(self):  # Prediction classes will be taken from 'pointInstance.get_point_class'
        # TODO
        pass


class Point:
    def __init__(self, list_of_values, point_class):
        self.list_of_values = list_of_values
        self.point_class = point_class

    def __str__(self):
        return "Values: " + str(self.get_point_values()) + ", Class: " + str(self.get_point_class())

    def get_point_value(self, number_of_value):
        if number_of_value > len(self.list_of_values):
            raise Exception("Too big number of value")
        elif number_of_value < 0:
            raise Exception("Too small number of value")
        else:
            return float(self.list_of_values[number_of_value])

    def get_point_values(self):
        return self.list_of_values

    def get_point_class(self):
        return self.point_class
