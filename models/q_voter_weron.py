from copy import copy

import networkx as nx
import numpy as np
from models import BaseQVoter


class QVoterWeron(BaseQVoter):
    """ q_a-voter model."""

    def __init__(self, init_network: nx.Graph):
        super().__init__(init_network=init_network)

    def single_step(self, p: float, q: int, type_of_influence: str = 'NN'):
        """ Single event accroding to the paper.
        Args:
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q (int): number of people in the influence group
            type_of_influence (str): type of choice of the influence group.
        """
        # (1) 'pick a spinson at random'
        spinson = np.random.choice(self.operating_network.nodes, 1)[0]

        # (2) 'decide with probability p, if the spinson will act as independent
        if np.random.random() < p:
            # (3) if independent, change it's opinion with probability 1/2
            if np.random.random() < 0.5:
                opinion = self.operating_opinion[spinson]
                self.operating_opinion[spinson] = -1 * opinion
        else:
            # (4) if not independent, let the spinson take the opinion of its randomly chosen group of influence.
            influence_group = self.influence_choice(spinson, q, type_of_influence)
            # only if the q_a-panel is unanimous
            if self.unanimous_check(influence_group):
                self.operating_opinion[spinson] = self.operating_opinion[list(influence_group)[0]]
            # TODO: there could be also a part from original model, but it's not part of this model:
            #       else: spinson flips it's opinion with probability <eps>.

    def simulate(self, num_of_events: int, p: float, q: int, type_of_influence: str = 'NN'):
        """ Method simulating the opinion spread: <num_of_events> steps.
        Args:
            num_of_events: number of iterations (time).
            p (flaot): 0 <= p <= 1. Probability for spinson to be independent
            q (int): number of people in the influence group
            type_of_influence (str): type of choice of the influence group.
        """
        self.initialize_simulation()

        for event in range(num_of_events):
            # single iteration
            self.single_step(p, q, type_of_influence)
            # add current magnetization to the list
            self.update_magnetization_list()

        return self.operating_magnetization

    def initialize_simulation(self, opinion_init: str = "all_positive", *args, **kwargs):
        """ Method initializing operating values, i.e. clearing them. """
        # cleaning operating network
        self.reload_operating_network()
        # cleaning operating opinion
        self.reload_operating_opinion()
        # cleaning magnetization
        self.reload_operating_magnetization()
