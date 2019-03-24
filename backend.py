import math
import random
import networkx as nx
import numpy as np

from operator import itemgetter
from enum import Enum, auto


class Point:
    """ Defines a Point class with attributes X, Y and
    utility functions to generate random points and find distances"""
    def __init__(self, x: float, y: float):
        """ Point is defined in 2D space with X, Y in [0, 1]"""
        assert 0 <= x <= 1 and 0 <= y <= 1
        self.X, self.Y = x, y

    @staticmethod
    def euclid_distance(p1, p2):
        """ Finds the distance between two given points """
        assert isinstance(p1, Point) and isinstance(p2, Point)
        return math.sqrt((p1.X - p2.X) ** 2 + (p1.Y - p2.Y) ** 2)

    @staticmethod
    def distance(p1, p2):
        return Point.euclid_distance(p1, p2)

    @staticmethod
    def random():
        """ Generates a random Point using uniform distribution """
        return Point(random.random(), random.random())

    def tuple(self):
        """ Converts the point to a tuple format (X, Y) """
        return self.X, self.Y

    @classmethod
    def from_tuple(cls, pair):
        """ Construct a Point object from the tuple(X, Y)"""
        assert len(pair) == 2
        return cls(pair[0], pair[1])

    @classmethod
    def normalize(cls, x, y, x_min, x_max, y_min, y_max):
        """ Construct a Point by normalizing the (X, Y) using given limits """
        assert x_max > x_min and y_max > y_min
        assert x_max >= x >= x_min and y_max >= y >= y_min
        return cls((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min))

    def __str__(self):
        return 'Point(x: %f,y: %f)' % (self.X, self.Y)


class NetworkGraph(nx.Graph):
    """ Defines a undirected graph class, with utility functions to generate connected graphs,
    with given number of paths between source and destination, calculate reduced paths, and
    functionality of generate layout for given graph (position of nodes to place while drawing)"""
    def __init__(self, num_nodes):
        nx.Graph.__init__(self)
        self.num_nodes = num_nodes
        self.start_node = 0
        self.end_node = num_nodes - 1

    def __randomize_points__(self):
        """Clear any existing points and assign new ones"""
        self.clear()
        for i in range(self.num_nodes):
            self.add_node(i, point=Point.random())

    def __calculate_distances__(self):
        """Find distances between points and sort potential edges (links) in increasing order of distances"""
        distances = [(Point.distance(self.nodes[i]['point'], self.nodes[j]['point']), i, j)
                     for i in range(self.num_nodes) for j in range(i)]
        self.links = list(map(itemgetter(1, 2), sorted(distances, key=itemgetter(0))))

    def __generate_graph__(self):
        """ Construct a connected graph with source and destination
        separated by multi-hop based on links obtained """
        for edge in self.links:
            if {self.start_node, self.end_node} == set(edge):
                continue
            self.add_edge(*edge)
            if nx.is_connected(self):
                break
        else:
            """ This should not execute for debugging in-case of unexpected behaviour"""
            raise RuntimeError('Unexpected in-ability to construct graph')

    def __calculate_paths__(self):
        """Construct reduced paths. A set of path is reduced if no path (in terms of nodes) is a subset of another """
        simple_paths = nx.all_simple_paths(self, self.start_node, self.end_node)
        self.reduced_paths = []
        for path in simple_paths:
            append_path = True
            for some_path in self.reduced_paths.copy():
                if set(path).issuperset(some_path):
                    append_path = False
                elif set(path).issubset(some_path):
                    self.reduced_paths.remove(some_path)
            if append_path:
                self.reduced_paths.append(path)

    def construct_connected_graph(self):
        """ Construct a connected graph by running required utility functions """
        self.__randomize_points__()
        self.__calculate_distances__()
        self.__generate_graph__()
        self.__calculate_paths__()

    def ensure_at_least_n_paths(self, n, retries=20):
        """ Construct a connected graph with at-least n reduced paths """
        for trial in range(retries):
            self.construct_connected_graph()
            if len(self.reduced_paths) >= n:
                return True
        """ If failed to construct graph with n paths after retrying """
        return False

    def calculate_layout_points(self):
        """ Use networkx layout to obtain points for drawing nodes """
        initial_positions = {i: self.nodes[i]['point'].tuple() for i in range(self.num_nodes)}
        spring_positions = nx.spring_layout(self, pos=initial_positions)
        """Normalize the spring positions and set them as layout points"""
        x_max, x_min = max(map(itemgetter(0), spring_positions.values())), min(
            map(itemgetter(0), spring_positions.values()))
        y_max, y_min = max(map(itemgetter(1), spring_positions.values())), min(
            map(itemgetter(1), spring_positions.values()))
        for i in range(self.num_nodes):
            self.nodes[i]['layout_point'] = Point.normalize(*spring_positions[i],
                                                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


class Model:
    """ Assumes that 0 is the start node and num_nodes - 1 is the end node and
    start node models the behaviour of adversarial nodes (remaining nodes)
    obeying the given assumptions specific to the model """
    @staticmethod
    def random_curiosity(num_nodes, **kwargs):
        raise NotImplementedError("random_curiosity not implemented")

    @staticmethod
    def random_collaboration(num_nodes, **kwargs):
        raise NotImplementedError("random_collaboration not implemented")

    @staticmethod
    def random_value(mu, std):
        """ Generate a random number between 0 and 1 with given mu and std from gaussian distribution """
        return min(1, max(0, random.gauss(mu=mu, sigma=std)))

    @staticmethod
    def objective_function(num_nodes, curiosity_matrix, collaboration_matrix):
        """ This function is expected to return a function which takes set of paths as input and
        returns the objective value"""
        raise NotImplementedError("objective_function not implemented")


class ProbabilisticModel(Model):
    class ObjectFunction(Enum):
        """ The Objective Function to use while computing objective value """
        """ Probability that no nodes break the secret """
        NO_NODE_BREAK_SECRET = auto()
        """ Max Probability of some node not breaking the secret """
        SOME_NODE_BREAK_SECRET = auto()

    @staticmethod
    def parse_gauss_params(kwargs):
        mu = kwargs['mu'] if 'mu' in kwargs else 0.5
        std = kwargs['std'] if 'std' in kwargs else 0.25
        return mu, std

    @staticmethod
    def random_curiosity(num_nodes, **kwargs):
        assert num_nodes >= 2
        mu, std = ProbabilisticModel.parse_gauss_params(kwargs)
        """ Source and destination are willing to put effort to break the secret """
        curiosity_matrix = np.ones(num_nodes)
        for i in range(1, num_nodes - 1):
            curiosity_matrix[i] = ProbabilisticModel.random_value(mu, std)
        return curiosity_matrix

    @staticmethod
    def random_collaboration(num_nodes, **kwargs):
        assert num_nodes >= 2
        mu, std = ProbabilisticModel.parse_gauss_params(kwargs)
        collaboration_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(0, num_nodes):
            """ Nodes always collaborate with themselves """
            collaboration_matrix[i, i] = 1.0
        """ Source and destination nodes do not collaborate with any other node """
        for i in range(1, num_nodes - 1):
            for j in range(1, i):
                """ Collaboration is mutual """
                mutual_collaboration = ProbabilisticModel.random_value(mu, std)
                collaboration_matrix[i, j] = collaboration_matrix[j, i] = mutual_collaboration
        return collaboration_matrix

    @staticmethod
    def objective_function(num_nodes, curiosity_matrix, collaboration_matrix):
        def objective_value(paths, obj_fn=ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET):
            """Each Share is indicated by the path it took"""
            num_shares = len(paths)
            gathering_probability = np.array(
                [np.prod([1 - np.prod([1 - collaboration_matrix[i, j] for j in path]) for path in paths]) for i in
                 range(num_nodes)])
            decoding_probability = np.array([curiosity_matrix[i] ** num_shares for i in range(num_nodes)])
            print('Gathering probability', ', '.join(map(str, gathering_probability)))
            print('Decoding probability',  ', '.join(map(str, decoding_probability)))
            breaking_probability = gathering_probability * decoding_probability
            print('Breaking probability', ', '.join(map(str, breaking_probability)))
            assert breaking_probability[0] == 1 and breaking_probability[num_nodes-1] == 1
            """ Source and destination node are not adversaries and do not contribute to objective value"""
            if obj_fn == ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET:
                """ Using the probability of nodes breaking, find the probability that no nodes break """
                return np.prod(1 - breaking_probability[1: -1])
                pass
            elif obj_fn == ProbabilisticModel.ObjectFunction.SOME_NODE_BREAK_SECRET:
                """ Find the maximum probability of some node breaking the secret and return the inverse """
                return 1 - max(breaking_probability[1: -1])
            else:
                raise NotImplementedError("ObjectiveFunction not yet implemented")
        return objective_value


if __name__ == '__main__':
    """ Just sample code to show usage """
    NUM_NODES = 12  # Including start and end points
    graph = NetworkGraph(num_nodes=NUM_NODES)
    graph.construct_connected_graph()
    print(graph.reduced_paths)

    graph.ensure_at_least_n_paths(n=5)
    print(graph.reduced_paths)

    graph.construct_connected_graph()
    graph.calculate_layout_points()
    for i in range(NUM_NODES):
        print('Original: ', graph.nodes[i]['point'], 'Layout: ', graph.nodes[i]['layout_point'])

    curiosity = ProbabilisticModel.random_curiosity(num_nodes=NUM_NODES)
    collaboration = ProbabilisticModel.random_collaboration(num_nodes=NUM_NODES)
    print('Curiosity:', curiosity)
    print('Collaboration:', collaboration)
    """ Sending a share from each of the reduced path 1 per path """
    paths = graph.reduced_paths
    objective_function = ProbabilisticModel.objective_function(num_nodes=NUM_NODES,
                                                               curiosity_matrix=curiosity,
                                                               collaboration_matrix=collaboration)
    print('Paths:', paths)
    print(ProbabilisticModel.ObjectFunction.SOME_NODE_BREAK_SECRET,
          objective_function(paths, ProbabilisticModel.ObjectFunction.SOME_NODE_BREAK_SECRET))
    print(ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET,
          objective_function(paths, ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET))
