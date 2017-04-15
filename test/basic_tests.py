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
for point in test.get_k_nearest_neighbors(3):
    print(point)

print("="*60)

for point in test.get_k_nearest_neighbors(3):
    print(point.get_point_value(1))

print("="*60)

for i in range(len(test.list_of_test_points)):
    neighbour = test.get_k_nearest_neighbors(i)
    print(test.get_prediction(neighbour))

print("="*60)

y_predicted = []
for i in range(len(test.list_of_test_points)):
    neighbour = test.get_k_nearest_neighbors(i)
    y_predicted.append(test.get_prediction(neighbour))

y_true = []
for point in test.list_of_test_points:
    y = point.get_point_class()
    y_true.append(y)

print("Score: %s" % test.calculate_accuracy(y_true, y_predicted))
print(y_true)
print(y_predicted)


