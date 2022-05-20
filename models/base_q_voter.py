import networkx as nx
from copy import copy
import numpy as np
from typing import Optional
from abs import abc
from matplotlib import pyplot as plt


@abc.abstractclass
class BaseQVoter:
    """ base q-voter model."""

    def __init__(self, init_network: nx.Graph):
        self.init_network = init_network
        self.network_size = init_network.number_of_nodes()

        self.operating_network = None
        self.operating_opinion = None
        self.operating_magnetization = []

    def reload_operating_network(self):
        """ Operating network is needed for Monte Carlo trajectories. """
        self.operating_network = copy(self.init_network)

    def reload_operating_opinion(self, init_type: str = "disordered_exact_fraction", *args, **kwargs):
        """ Method initializing opinion of the spinsons to 1. In future this could be changed and improved. """
        if init_type == "disordered_exact_fraction":
            self.operating_opinion = self.create_opinion_exact_fraction(*args, **kwargs)
        elif init_type == "p_for_positive":
            self.operating_opinion = self.create_opinion_according_to_p(*args, **kwargs)
        elif init_type == "all_positive":
            self.operating_opinion = {node: 1 for node in self.init_network.nodes}
        elif init_type == "all_negative":
            self.operating_opinion = {node: -1 for node in self.init_network.nodes}
        else:
            raise NotImplementedError

    def create_opinion_according_to_p(self, c: float = 0.5) -> dict:
        """ Function creating vector of opinions according to p for 1 """
        return {node: np.random.choice((-1, 1), p=(c, 1-c)) for node in self.init_network.nodes}

    def create_opinion_exact_fraction(self, c: float = 0.5) -> dict:
        """ Function creating vector of opinions according to p for 1 """
        frac = round(self.network_size * c)

        positive_nodes = np.random.choice(self.init_network.nodes, frac, replace=False)

        positive_opinions = {node: 1 for node in positive_nodes}
        negative_opinions = {node: -1 for node in self.init_network.nodes if node not in positive_nodes}

        # FYI from Python 3.9 '|' operator merges 2 dictionaries. Doesn't work for lists :/
        return positive_opinions | negative_opinions

    def reload_operating_magnetization(self):
        self.operating_magnetization = []

    def influence_choice(self, spinson: int, q: int, type_of_influence: str = 'RND_no_repetitions') -> list:
        """ Method returning spinsons from the network to affect given <spinson (int)> according to given theoretical
            <type_of_influence (int)>
        Args:
            spinson (int):  given spinson.
            q (int): number of people in the influence group
            type_of_influence (str): type of choice of the influence group.
        """

        if type_of_influence == 'RND_no_repetitions':
            # 'q randomly chosen nearest neighbours of the target spinson are in the group. No repetitions.'
            neighbours = [neighbour for neighbour in self.operating_network.neighbors(spinson)]
            if len(neighbours) < q:
                return neighbours
            else:
                return np.random.choice(neighbours, q, replace=False)
        elif type_of_influence == 'NN':
            # 'q randomly chosen nearest neighbours of the target spinson are in the group.'
            return np.random.choice([neighbour for neighbour in self.operating_network.neighbors(spinson)], q)
        else:
            # in the future there may be other ways of choice implemented as well
            raise NotImplementedError

    def unanimous_check(self, group: list[int]):
        """ Method checking if the group is unanimous.
        Args:
            group (list[int]): Given group"""
        # only if (all are equal to 1) v (all are equal to -1)  <==> abs(sum(group_opinions)) = len(group)
        opinions = [self.operating_opinion[member] for member in group]
        return abs(sum(opinions)) == len(group)

    def single_step(self, *args, **kwargs):
        """ Single event. """
        raise NotImplementedError

    def simulate(self, *args, **kwargs) -> list:
        """ Method simulating the opinion spread: <num_of_events> steps. """
        pass

    def simulate_until_stable(self, *args, **kwargs) -> tuple[list, int, float]:
        """ Method simulating the opinion spread. Simulates it until <max_iterations> iterations. From min_iterations
            algorithm compares actual value with MA(<ma_value>) previous values*. MA stands for moving average.
            For optimization purposes, check is done once a <ma_value_iterations>
        * it does that by checking variance of these Values. If variance is equal to 0, loop breaks.

        Args:
            min_iterations (int): minimal number of iterations
            ma_value (int): number of values to compare with actual value
            eps (int): difference between actual value and (moving) average of <ma_value> previous values
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q_a (int): number of people in the influence group for independent spinson (anticonformity case)
            q_c (int): number of people in the influence group for not independent spinson (conformity case)
            type_of_influence (str): type of choice of the influence group.
            min_iterations (Optional[int]): maximal number of iterations
        Returns:
            (list): magnetization
            (int): number of iterations
            (float): global concentration
        """
        pass

    def calculate_global_concentration(self):
        """ Method calculating global concentration. Positive/all"""
        return len([opinion for opinion in self.operating_opinion.values() if opinion == 1])/len(self.operating_opinion)

    def calculate_magnetization(self):
        """ Method calculating magnetization. """
        return np.mean(list(self.operating_opinion.values()))

    def update_magnetization_list(self):
        """ Method updating magnetization list with current magnetization. """
        self.operating_magnetization.append(self.calculate_magnetization())

    def initialize_simulation(self, opinion_init: str = "disordered_exact_fraction", *args, **kwargs):
        """ Method initializing operating values, i.e. clearing them. """
        raise NotImplementedError
