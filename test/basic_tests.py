from lib.KNearestNeighbors import *

test = KNearestNeighbors()
test.load_data_from_csv("./data/iris_data.txt")

# Proper loading test
for i in range(len(test.list_of_points)):
    point = test.list_of_points[i]
    print("Points: %s | classes: %s" % (point.get_point_values(), point.get_point_class()))
