import csv


class KNearestNeighbors:

    def __init__(self):
        self.list_of_points = []

    def load_data_from_csv(self, path):
        with open(path, mode="rt", encoding="utf-8") as file:  # Open file with text mode
            rows = csv.reader(file)
            for row in rows:
                point_values = []
                if len(row) == 0:  # Don't create the Point instance if the row is empty
                    continue
                for number_of_value in range(len(row)-1):  # extract values
                    point_values.append(row[number_of_value])
                point_class = row[len(row)-1]
                self.list_of_points.append(Point(point_values, point_class))

    def calculate_euclidean_distance(self, point_1, point_2):
        pass

    def get_k_nearest_neighbors(self, k):
        pass

    def get_response(self):
        pass

    def calculate_accuracy(self):
        pass


class Point:
    def __init__(self, list_of_values, point_class):
        self.list_of_values = list_of_values
        self.point_class = point_class

    def get_point_value(self, number_of_value):
        if number_of_value > len(self.list_of_values):
            raise Exception("Too big number of value")
        elif number_of_value < 0:
            raise Exception("Too small number of value")
        else:
            return self.list_of_values[number_of_value]

    def get_point_values(self):
        return self.list_of_values

    def get_point_class(self):
        return self.point_class
