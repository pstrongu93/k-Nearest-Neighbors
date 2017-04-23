# -*- coding: utf-8 -*-
import unittest
from lib.KNearestNeighbors import KNearestNeighbors
from lib.KNearestNeighbors import Point
class TestKNearestNeighbors(unittest.TestCase):
    """
    Class for unit testing
    """
    def testKNearestNeighborsExist(self):
        """
        method for checking class existence 
        :param TestKNearestNeighbors self: class instance
        """
        self.KNearestNeighbors = KNearestNeighbors()

    def testPointExist(self):
        """
        method for checking class existence 
        :param TestKNearestNeighbors self: class instance
        """
        self.Point = Point([],'')

    def testKNearestNeighborsHasListOfTrainingPoints(self):
        """
        method for checking 'list of training points' class attribute
        :param TestKNearestNeighbors self:  class instance
        """
        self.KNearestNeighbors = KNearestNeighbors()
        assert self.KNearestNeighbors.list_of_training_points != None, "list_of_training_points is None"

    def testKNearestNeighborsHasListOfTestPoints(self):
        """
        method for checking 'list of test points class' attribute
        :param TestKNearestNeighbors self:  class instance
        """
        self.KNearestNeighbors = KNearestNeighbors()
        assert self.KNearestNeighbors.list_of_test_points != None, "list_of_test_points is None"

    def testKNearestNeighborsHasKValue(self):
        """
        method for checking 'k' class attribute
        :param TestKNearestNeighbors self:  class instance
        """
        self.KNearestNeighbors = KNearestNeighbors()
        assert self.KNearestNeighbors.k != None, "k is None"

    def testPointHasListOfValues(self):
        """
        method for checking 'list_of_values' Point class attribute
        :param TestKNearestNeighbors self:  class instance
        """
        self.Point = Point([],'')
        assert self.Point.list_of_values != None, "list_of_values is None"

    def testPointHasPointClass(self):
        """
        method for checking 'point_class' Point class attribute
        :param TestKNearestNeighbors self:  class instance
        """
        self.Point = Point([],'')
        assert self.Point.point_class != None, "point_class is None"

    def testGetPointClass(self):
        """
        method for checking method get_point_class(), which should return point class
        :param TestKNearestNeighbors self:  class instance
        """
        self.Point = Point([],'classA')
        assert self.Point.get_point_class() == 'classA', "get point class method isn't work propertly"

    def testGetPointValues(self):
        """
        method for checking method get_point_values(), which should return list of points
        :param TestKNearestNeighbors self:  class instance
        """
        self.Point = Point(['1.0','1.1','1,2'],'')
        assert self.Point.get_point_values() == ['1.0','1.1','1,2'], "get point values method isn't work propertly"

    def testGetPointValue(self):
        """
        method for checking method get_point_value(), which should return one point value by index
        :param TestKNearestNeighbors self:  class instance
        """
        self.Point = Point(['1.0','1.1','1.2'],'')
        assert self.Point.get_point_value(2) == 1.2 , "get point value method isn't work propertly"

    def testGetPointValueOutOfBound(self):
        """
        method for checking out of array bound exception for get_point_value() 
        :param TestKNearestNeighbors self:  class instance
         """
        with self.assertRaises(Exception) as context:
            self.Point = Point(['1.0', '1.1', '1.2'], '')
            self.Point.get_point_value(3)
        self.assertFalse("Too big number of value" in str(context.exception))

    def testGetPointValueSmallerThanZero(self):
        """
        method for checking exception when method argument in smaller than 0
        :param TestKNearestNeighbors self:  class instance
        """
        with self.assertRaises(Exception) as context:
            self.Point = Point(['1.0', '1.1', '1.2'], '')
            self.Point.get_point_value(-1)
        self.assertTrue("Too small number of value" in str(context.exception))

    def testKNearestNeighborsCalculateAccuracy(self):
        """
        method for checking correctness of calculate accuracy method
        :param TestKNearestNeighbors self:  class instance
        """
        self.KNearestNeighbors = KNearestNeighbors()
        assert self.KNearestNeighbors.calculate_accuracy([1,2],[1,2]) == 100, "Accuracy calculator method not working propertly"

    def testKNearestNeighborsCalculateEuclideanDistance(self):
        """
        method for checking correctness of calculate eucliedan distance method
        :param TestKNearestNeighbors self:  class instance
        """
        self.KNearestNeighbors = KNearestNeighbors()
        self.Point1 = Point([],'')
        self.Point1.list_of_values = ['-2.0','2.0']
        self.Point2 = Point([], '')
        self.Point2.list_of_values = ['2.0', '-1.0']
        assert self.KNearestNeighbors.calculate_euclidean_distance(self.Point1, self.Point2) == 5, "Euclidan distance calculator method working not propertly"

    def testGetPrediction(self):
        """
        method for checking correctness of choose the class with higest value
        :param TestKNearestNeighbors self:  class instance
        """
        self.KNearestNeighbors = KNearestNeighbors()
        points = [Point([],"Iris-virginica"),  Point([], "Iris-setosa"),  Point([], "Iris-versicolor"), Point([],"Iris-virginica")]
        assert self.KNearestNeighbors.get_prediction(points) == "Iris-virginica", "GetPrediction is not working correctly"

    def testTraningData(self):
        """
        method for checking existence of wile with traning data
        :param TestKNearestNeighbors self:  class instance
        """
        raised = False
        try:
            path_to_training_data = "../data/training.txt"
            self.KNearestNeighbors = KNearestNeighbors()
            self.KNearestNeighbors.load_data_from_csv(path_to_training_data, training=True)
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised - there is no file in this path')

    def testTestData(self):
        """
        method for checking existence of wile with test data
        :param TestKNearestNeighbors self:  class instance
        """
        raised = False
        try:
            path_to_test_data = "../data/test.txt"
            self.KNearestNeighbors = KNearestNeighbors()
            self.KNearestNeighbors.load_data_from_csv(path_to_test_data, training=False)
        except:
            raised = True
        self.assertFalse(raised, 'Exception raised - there is no file in this path')






if __name__ == "__main__":
    unittest.main()



