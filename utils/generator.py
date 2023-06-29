import os
import random

import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import const_define as cd


class StrwaiDataGenerator:

    def __init__(self,
                 labelsAI: dict,
                 countries: dict,
                 q_rank: tuple = None,
                 r_rank: tuple = None,
                 q_countries: tuple = None,
                 r_countries: tuple = None,
                 seed: int = 424242):
        """
        StairwAI-like data generator

        :param labelsAI: dict of AI techniques <labelID>:<labelName>
        :param countries: dict of countries <countryID>:<countryName>
        :param q_rank: rank of AI techniques in the queries
        :param r_rank: rank of AI techniques in the resources
        :param q_countries: percentage of country in the queries
        :param r_countries: percentage of country in the resources
        :param seed: reproducibility seed
        """
        # Labels characterizing data
        self.labelsAI = labelsAI
        self.countries = countries
        self.n_labelsAI = len(labelsAI)
        self.n_countries = len(countries)

        # Fix seed for reproducibility
        self.seed = seed
        self.set_seed(self.seed)

        # Rank labelsAI for queries and resources
        self.q_rank = q_rank
        self.r_rank = r_rank
        assert self.n_labelsAI == len(
            self.q_rank) if self.q_rank else True, f"Len of q_rank ({len(self.q_rank)}) doesn't match # of labelsAI ({self.n_labelsAI})"
        assert self.n_labelsAI == len(
            self.r_rank) if self.r_rank else True, f"Len of r_rank ({len(self.r_rank)}) doesn't match # of labelsAI ({self.n_labelsAI})"

        # Percentage of country unbalance in queries and resources
        self.q_countries = tuple([1 / self.n_countries for _ in self.countries]) if q_countries is None else q_countries
        self.r_countries = tuple([1 / self.n_countries for _ in self.countries]) if r_countries is None else r_countries

    def rank_labels(self):
        """Sample a random rank for the labelsAI"""
        # Sample a random rank
        rank = random.sample(population=range(self.n_labelsAI), k=self.n_labelsAI)
        return tuple(rank)

    def generate_queries(self, n_queries: int):
        """Generate n_queries instances of query
        :param n_queries: number of queries to generate
        :return: array of queries
        """
        # Generate query scores for AI labels
        queriesAI = np.zeros((n_queries, self.n_labelsAI))
        for i in range(n_queries):
            if self.q_rank:
                queriesAI[i] = self.power_law(self.q_rank)
            else:
                queriesAI[i] = np.random.uniform(0., 1.)
        # Generate discrete query scores for countries
        queriesCountry = self.country_score(n_queries, self.q_countries)
        # Concatenate scores
        queries = np.concatenate([queriesAI, queriesCountry], axis=-1)
        return queries

    def generate_resources(self, n_resources: int):
        """Generate n_resources instances of resource
        :param n_resources: number of resources to generate
        :return: array of resources
        """
        # Generate query scores for AI labels
        resourcesAI = np.zeros((n_resources, self.n_labelsAI))
        for i in range(n_resources):
            if self.r_rank:
                resourcesAI[i] = self.power_law(self.r_rank)
            else:
                resourcesAI[i] = np.random.uniform(0., 1.)
        # Generate discrete query scores for countries
        resourcesCountry = self.country_score(n_resources, self.r_countries)
        # Concatenate scores
        resources = np.concatenate([resourcesAI, resourcesCountry], axis=-1)
        return resources

    def generate_data(self, n_queries: int, n_resources: int):
        """Generate queries and resources
        :param n_queries: number of queries to generate
        :param n_resources: number of resources to generate
        :return: array of queries, array of resources
        """
        # Generate queries
        queries = self.generate_queries(n_queries)
        # Generate resources
        resources = self.generate_resources(n_resources)
        return queries, resources

    def set_seed(self, seed):
        """
        Fix seed for reproducibility
        :param seed: seed to be fixed
        :return: fixed seed
        """
        seed = cd.set_seed(seed)
        return seed

    def __str__(self):
        s = f"Labels:\n {self.labelsAI}\n  with q_rank:\n {self.q_rank}\n  and r_rank:\n {self.r_rank}\n"
        s += f"  with q_countries:\n {self.q_countries}\n  and r_countries:\n {self.r_countries}"
        return s

    def power_law(self, rank: tuple):
        """Compute power law factors based on rank
        :param rank: tuple of rank
        :return: array of power low factors
        """
        factors = np.ones((self.n_labelsAI,))
        exp = np.zeros((self.n_labelsAI,))
        # Interval for the exponents
        sc = MinMaxScaler((1, 50))
        for i in range(self.n_labelsAI):
            # Label position in the q_rank
            idx = rank.index(i)
            # Exponent
            exp[i] = self.n_labelsAI - idx
        # Scale exponents
        scaled_exp = sc.fit_transform(exp.reshape(-1, 1)).reshape(-1, )
        factors *= np.random.power(scaled_exp)
        return factors

    def country_score(self, n_queries, q_countries):
        countries = np.random.multinomial(1, q_countries, size=n_queries)
        return countries

    def plot_data(self, queries: np.array, resources: np.array, save=False, name='distr'):
        """Plot data and resources distribution
        :param queries: array of queries
        :param resources: array of resources
        :param save: whether to save the plot
        :param name: name of image to save
        """

        fontsize = 16

        ax = plt.figure(figsize=(9, 4), dpi=130).add_subplot(111)

        # Resources Distr
        v = plt.violinplot(resources, showmeans=True)
        # Violin plot custom colors to mimic transparency
        for pc in v['bodies']:
            pc.set_facecolor('#aacbf5')
            pc.set_edgecolor('#aacbf5')
            pc.set_alpha(1)
        plt.vlines(9.55, -0.1, 1.1, color='black', linestyle='dotted')
        plt.title(f'Resources', )
        plt.ylim(-0.1, 1.1, )
        plt.ylabel('Scores', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(range(1, self.n_labelsAI + self.n_countries + 1),
                   [self.labelsAI[i] for i in range(self.n_labelsAI)] + [self.countries[i] for i in
                                                                         range(self.n_labelsAI,
                                                                               self.n_labelsAI + self.n_countries)],
                   rotation=50, fontsize=fontsize)
        for i in range(-1, -6, -1):
            ax.get_xticklabels()[i].set_weight("bold")

        if save:
            plt.savefig(os.path.join(cd.PROJECT_DIR, name + '_Res.eps'), format='eps', bbox_inches="tight")
        plt.show()

        ax = plt.figure(figsize=(9, 4), dpi=130).add_subplot(111)
        # Query Distr
        v = plt.violinplot(queries, showmeans=True)
        # Violin plot custom colors to mimic transparency
        for pc in v['bodies']:
            pc.set_facecolor('#aacbf5')
            pc.set_edgecolor('#aacbf5')
            pc.set_alpha(1)
        plt.vlines(9.55, -0.1, 1.1, color='black', linestyle='dotted')
        plt.title(f'Queries')
        plt.ylim(-0.1, 1.1, )
        plt.ylabel('Scores', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(range(1, self.n_labelsAI + self.n_countries + 1),
                   [self.labelsAI[i] for i in range(self.n_labelsAI)] + [self.countries[i] for i in
                                                                         range(self.n_labelsAI,
                                                                               self.n_labelsAI + self.n_countries)],
                   rotation=50, fontsize=fontsize)
        for i in range(-1, -6, -1):
            ax.get_xticklabels()[i].set_weight("bold")

        if save:
            plt.savefig(os.path.join(cd.PROJECT_DIR, name + '_Query.eps'), format='eps', bbox_inches="tight")
        plt.show()
