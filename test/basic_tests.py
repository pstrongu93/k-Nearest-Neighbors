from lib.KNearestNeighbors import *

test = KNearestNeighbors()

# Proper loading test
for i in range(len(test.list_of_training_points)):
    point = test.list_of_training_points[i]
    print("Points: %s | classes: %s" % (point.get_point_values(), point.get_point_class()))

point_1 = test.list_of_training_points[1]
point_2 = test.list_of_training_points[2]

distance = test.calculate_euclidean_distance(point_1, point_2)
print(distance)

