from KNearestNeighbors import *

test = KNearestNeighbors()
test.load_data_from_csv("./iris_data.txt")

for i in range(len(test.list_of_points)):
    point = test.list_of_points[i]
    print("Punkty: %s | klasa: %s" % (point.get_point_values(), point.get_point_class()))
