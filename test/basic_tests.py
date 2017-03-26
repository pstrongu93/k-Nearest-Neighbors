from lib.KNearestNeighbors import *

test = KNearestNeighbors()

print("="*60)
print("Proper loading test")
for i in range(len(test.list_of_training_points)):
    point = test.list_of_training_points[i]
    print("Points: %s | classes: %s" % (point.get_point_values(), point.get_point_class()))

print("="*60)
print("Distance test")
point_1 = test.list_of_training_points[1]
point_2 = test.list_of_training_points[2]
distance = test.calculate_euclidean_distance(point_1, point_2)
print(distance)

print("="*60)
print("K-nearest neighbors test")
for point in test.get_k_nearest_neighbors(2):
    print(point)

