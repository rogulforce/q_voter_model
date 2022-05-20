import networkx as nx
from copy import copy
import numpy as np
from typing import Optional

from matplotlib import pyplot as plt

from models.base_q_voter import BaseQVoter


class QVoterNature(BaseQVoter):
    """ q_a-voter model with NN influence group. """

    def __init__(self, init_network: nx.Graph):
        super().__init__(init_network=init_network)

    def single_step(self, p: float, q_a: int, q_c: int, type_of_influence: str = 'RND_no_repetitions'):
        """ Single event. According to the paper: https://www.nature.com/articles/s41598-021-97155-0
        Args:
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q_a (int): number of people in the influence group for independent spinson (anticonformity case)
            q_c (int): number of people in the influence group for not independent spinson (conformity case)
            type_of_influence (str): type of choice of the influence group.
        """
        # (1) 'pick a spinson at random'
        spinson = np.random.choice(self.operating_network.nodes, 1)[0]

        # (2) 'decide with probability p, if the spinson will act as independent
        if np.random.random() < p:
            # (3) if independent, change it's opinion (opposite of the group opinion)
            influence_group = self.influence_choice(spinson, q_a, type_of_influence)
            # only if the q_a-panel is unanimous
            if self.unanimous_check(influence_group):
                self.operating_opinion[spinson] = -1 * self.operating_opinion[list(influence_group)[0]]
        else:
            # (4) if not independent, let the spinson take the opinion of its randomly chosen group of influence.
            influence_group = self.influence_choice(spinson, q_c, type_of_influence)
            # only if the q_a-panel is unanimous
            if self.unanimous_check(influence_group):
                self.operating_opinion[spinson] = self.operating_opinion[list(influence_group)[0]]

    def simulate_until_stable(self, min_iterations: int, ma_value: int, p: float, q_a: int, q_c,
                              type_of_influence: str = 'RND_no_repetitions',
                              max_iterations: Optional[int] = 10 ** 5, opinion_init: str = "disordered_exact_fraction",
                              *args, **kwargs) -> tuple[list, int, float]:
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
        self.initialize_simulation(opinion_init, *args, **kwargs)

        num_of_iter = 0
        iter_from_last_check = 0
        while True:
            num_of_iter += 1
            iter_from_last_check += 1
            # single iteration
            self.single_step(p, q_a, q_c, type_of_influence)
            # add current magnetization to the list
            self.update_magnetization_list()

            # check for break
            if num_of_iter > min_iterations:
                if iter_from_last_check > ma_value:
                    if self.operating_magnetization[-1] == self.operating_magnetization[-2] and \
                            np.var(self.operating_magnetization[-1 * ma_value:]) == 0:
                        break
                    iter_from_last_check = 0

            if num_of_iter >= max_iterations:
                break
        concentration = self.calculate_global_concentration()
        return self.operating_magnetization, num_of_iter, concentration

    def initialize_simulation(self, opinion_init: str = "disordered_exact_fraction", *args, **kwargs):
        """ Method initializing operating values, i.e. clearing them. """
        # cleaning operating network
        self.reload_operating_network()
        # cleaning operating opinion
        self.reload_operating_opinion(opinion_init, *args, **kwargs)
        # cleaning magnetization
        self.reload_operating_magnetization()
