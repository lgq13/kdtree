from typing import List
from collections import namedtuple
import time
from matplotlib import pyplot as plt


class Point(namedtuple("Point", "x y")):  # representing the coordinate in 2-dimention plane
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'


class Rectangle(namedtuple("Rectangle", "lower upper")):
    #  the left-lower Point and right-upper Point, representing the range of the rectangle
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'

    def is_contains(self, p: Point) -> bool:
        # determine the input Point instance is contain in the Rectangle instance
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y


class Node(namedtuple("Node", "location left right")):
    # coordinate information of this Node, left-child Node, right-child Node
    """
    location: Point
    left: Node
    right: Node
    """

    def __repr__(self):
        return f'{tuple(self)!r}'


class KDTree:
    """k-d tree"""

    def __init__(self):
        self._root = None  # initialize root to None
        self._n = 0  # initialize depth to 0

    def insert(self, p: List[Point]):
        """insert a list of points"""

        def rec_ins(lst: List[Point], depth: int):  # a recursive function to create Nodes
            if not lst:
                return None
            axis = depth % 2  # determine split axis by depth % 2 method
            mid = len(lst) // 2  # determine the index of median
            if axis == 0:  # split by x axis
                lst.sort(key=lambda pt: pt.x)  # Sort in ascending order according to x coordinate
            else:  # split by y axis
                lst.sort(key=lambda pt: pt.y)  # Sort in ascending order according to y coordinate
            left_lst = lst[:mid]  # list of left child Points
            right_lst = lst[mid + 1:]  # list of right child Points
            return Node(lst[mid], rec_ins(left_lst, depth + 1), rec_ins(right_lst, depth + 1))  # insert recursively

        self._root = rec_ins(p, self._n)  # start the recursive function from root Node

    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""
        result = []  # creat a list contains the result

        def rec_ran(rec: Rectangle, node: Node, depth: int):  # recursive range query function
            if not node:
                return None
            axis = depth % 2  # determine split axis
            if axis == 0:  # split by x axis
                if node.location.x < rec.lower.x:  # x coordinate < range, search right-child tree in next step
                    rec_ran(rec, node.right, depth + 1)
                elif node.location.x > rec.upper.x:  # x coordinate > range, search left-child tree in next step
                    rec_ran(rec, node.left, depth + 1)
                else:  # x coordinate within range, search both right-child and left-child tree in next step
                    rec_ran(rec, node.right, depth + 1)
                    rec_ran(rec, node.left, depth + 1)
                    if rec.is_contains(node.location):  # determine the position of the node is in the rectangle
                        result.append(node.location)  # append it into the result list if it is

            else:  # split by y axis
                if node.location.y < rec.lower.y:  # y coordinate < range, search right-child tree in next step
                    rec_ran(rec, node.right, depth + 1)
                elif node.location.y > rec.upper.y:  # y coordinate > range, search left-child tree in next step
                    rec_ran(rec, node.left, depth + 1)
                else:  # y coordinate within range, search both right-child and left-child tree in next step
                    rec_ran(rec, node.right, depth + 1)
                    rec_ran(rec, node.left, depth + 1)
                    if rec.is_contains(node.location):  # determine the position of the node is in the rectangle
                        result.append(node.location)  # append it into the result list if it is

        rec_ran(rectangle, self._root, self._n)  # start recursive process form root node
        return result

    def knn(self, target_pt: Point):  # nearest node query
        global dis
        global path
        global n_node
        global last_node
        global checked_node
        path = []  # record query path
        n_node = None  # current nearest node
        dis = 0  # disstance between current nearest node and target node

        def distance(point: Point):  # distance calculation function
            return ((point.x - target_pt.x) ** 2 + (point.y - target_pt.y) ** 2) ** (0.5)

        def nearest_leaf(node: Node, depth: int):  # search for the nearest leaf node
            if node is None:
                return None
            path.append(node)
            if node.left is None and node.right is None:
                return None
            axis = depth % 2
            if axis == 0:
                if node.location.x <= target_pt.x:
                    nearest_leaf(node.right, depth + 1)
                else:
                    nearest_leaf(node.left, depth + 1)
            else:
                if node.location.y <= target_pt.y:
                    nearest_leaf(node.right, depth + 1)
                else:
                    nearest_leaf(node.left, depth + 1)

        nearest_leaf(self._root, 0)
        n_node = path.pop()
        dis = distance(n_node.location)
        last_node = n_node  # record the last node
        checked_node = []  # define a checked node list, avoid to check a child node twice
        checked_node.append(last_node)

        def trace_back():  # trace back through the path, to see if there is any node has shorter distance
            global dis
            global path
            global n_node
            global last_node
            global checked_node

            if not len(path):
                return None
            current_node = path[-1]
            if dis <= distance(
                    current_node.location):  # if the distance to current node > the distance to current nearest node
                path.pop()
                last_node = current_node
                checked_node.append(last_node)
                trace_back()
            else:  # if the distance to current node < the distance to current nearest node
                dis = distance(current_node.location)
                n_node = current_node
                # downward search another unchecked child tree of current node to see if there is and node has shorter distance
                if sorted([last_node]) == sorted([current_node.left]):
                    if current_node.right not in checked_node and current_node.right is not None:
                        nearest_leaf(current_node.right, len(path))
                        trace_back()
                elif sorted([last_node]) == sorted([current_node.right]):
                    if current_node.left not in checked_node and current_node.left is not None:
                        nearest_leaf(current_node.left, len(path))
                        trace_back()
                if len(path) != 0:
                    path.pop()
                    last_node = current_node
                    checked_node.append(last_node)
                    trace_back()

        trace_back()
        return n_node.location


def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    print(f'Naive method: {end - start}ms')

    kd = KDTree()
    kd.insert(points)
    # k-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    print(f'K-D tree: {end - start}ms')

    assert sorted(result1) == sorted(result2)


def visualize_performance():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]
    kd = KDTree()
    kd.insert(points)

    x = []
    y_naive = []
    y_kd = []

    for i in range(0, 1200, 3):
        x.append(i)
        lower = Point(0, 0)
        upper = Point(i, i)
        rectangle = Rectangle(lower, upper)
        #  naive method
        start = int(round(time.time() * 1000))
        result1 = [p for p in points if rectangle.is_contains(p)]
        end = int(round(time.time() * 1000))
        y_naive.append(end - start)

        # k-d tree
        start = int(round(time.time() * 1000))
        result2 = kd.range(rectangle)
        end = int(round(time.time() * 1000))
        y_kd.append(end - start)

    plt.figure(figsize=(18, 8), dpi=(100))
    plt.ylabel("time : ms", fontdict={'size': 16})
    plt.plot(x, y_kd, c='red', label='kd-tree')
    plt.plot(x, y_naive, c='black', label='naive method')
    plt.legend(loc='best')
    plt.show()


def knn_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = [kd.knn(Point(2.1, 3.1))]
    assert sorted(result) == sorted([Point(2, 3)])
    print('{}是距离Point(2.1, 3.1)最近的点'.format(result))
    result = [kd.knn(Point(2, 4.5))]
    assert sorted(result) == sorted([Point(2, 3)])
    print('{}是距离Point(2, 4.5)最近的点'.format(result))


if __name__ == '__main__':
    range_test()
    performance_test()
    knn_test()
    # visualize_performance()  # It will take about 4 minutes.
