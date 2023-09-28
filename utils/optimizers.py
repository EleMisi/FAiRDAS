import numpy as np

import const_define as cd


class RandomWalk:
    def __init__(self, max_iter: int, n_samples: int, decay_rate: float, cost_fn, tolerance: float, patience: int,
                 seed: int, verbose: bool = True):
        """
        Random walk algorithm class
        :param max_iter:
            maximum number of iterations
        :param n_samples:
            number of new points to evaluate
        :param decay_rate:
            decay rate for flipping probability
        :param cost_fn:
            objective function to minimize
        :param tolerance:
            improvement threshold for stopping criterion
        :param patience:
            number of iterations with no improvement for stopping criterion
        :param seed:
            seed for reproducibility
        """
        self.max_iter = max_iter
        self.cost_fn = cost_fn
        self.n_samples = n_samples
        self.decay_rate = decay_rate
        self.tolerance = tolerance
        self.patience = patience
        self.seed = self.set_seed(seed)
        self.verbose = verbose

    def __call__(self, flag: np.ndarray, args: dict = {}):
        """
        Perform random walk with diminishing flipping probability starting from the given set of weigths
        :param flag:
            matrix with starting weights
        :param args:
            cost function parameters
        :return:
            tuple with visited weights, corresponding costs and flipping probs
        """
        weight_history = [flag]
        # Current state cost
        curr_cost, x_approx, new_rank = self.cost_fn(flag, **args)
        cost_history = [[curr_cost, x_approx, new_rank]]
        flipping_p_history = [(1 - self.decay_rate)]

        tol = self.patience
        for k in range(1, self.max_iter + 1):

            # Diminish flipping prob
            flipping_p = (1 - self.decay_rate) ** k

            # Compute new candidates point according to the flipping probability
            flip_masks = np.array(
                np.random.choice([1, 0], p=[flipping_p, 1 - flipping_p], size=(self.n_samples, *flag.shape)),
                dtype=bool)
            w_candidates = np.logical_not(np.repeat(flag[None, :], self.n_samples, axis=0), where=flip_masks,
                                          out=np.repeat(flag[None, :], self.n_samples, axis=0).copy())

            # Evaluate all candidates
            evals_list = [self.cost_fn(w_val, **args) for w_val in w_candidates]
            evals_cost = np.array([e[0] for e in evals_list])
            evals_x_approx = np.array([e[1] for e in evals_list])
            evals_new_rank = np.array([e[2] for e in evals_list])

            del evals_list

            # Search for descent direction
            ind = np.argmin(evals_cost)
            min_cost = evals_cost[ind]
            if min_cost < curr_cost:
                # Update tolerance
                if curr_cost - min_cost < self.tolerance:
                    tol -= 1
                else:
                    tol = self.patience
                # Update weights
                flag = w_candidates[ind]
                curr_cost = min_cost
                x_approx = evals_x_approx[ind]
                new_rank = evals_new_rank[ind]
            else:
                tol -= 1
            # Check patience stopping criterion
            if tol == 0:
                if self.verbose:
                    print(f"Reached patience stopping criterion at iteration {k}")
                break
            # Record weights, cost evaluation anf flipping prob
            weight_history.append(flag)
            cost_history.append([curr_cost, x_approx, new_rank])
            flipping_p_history.append(flipping_p)
        return weight_history, cost_history, flipping_p_history, k

    def set_seed(self, seed):
        """
        Fix seed for reproducibility
        :param seed: seed to be fixed
        :return: fixed seed
        """
        seed = cd.set_seed(seed)
        return seed


class RandomWalk_wHistory(RandomWalk):

    def __init__(self, max_iter: int, n_samples: int, decay_rate: float, cost_fn, tolerance: float, patience: int,
                 seed: int, verbose: bool = True):
        """
        Random walk algorithm class
        :param max_iter:
            maximum number of iterations
        :param n_samples:
            number of new points to evaluate
        :param decay_rate:
            deacay rate for flipping probability
        :param cost_fn:
            objective function to minimize
        :param tolerance:
            improvement threshold for stopping criterion
        :param patience:
            number of iterations with no improvement for stopping criterion
        :param seed:
            seed for reproducibility
        """
        super(RandomWalk_wHistory, self).__init__(max_iter, n_samples, decay_rate, cost_fn, tolerance, patience,
                                                  seed, verbose)

    def __call__(self, flag: np.ndarray, args: dict = {}):
        """
        Perform random walk with diminishing flipping probability starting from the given set of weigths
        :param flag:
            matrix with starting weights
        :param args:
            cost function parameters
        :return:
            tuple with visited weights, corresponding costs and flipping probs
        """
        weight_history = [flag]
        # Current state cost with hystory
        curr_cost, x_approx, new_rank = self.cost_fn(flag, **args)
        cost_history = [[curr_cost, x_approx, new_rank]]
        flipping_p_history = [(1 - self.decay_rate)]

        tol = self.patience
        for k in range(1, self.max_iter + 1):

            # Diminish flipping prob
            flipping_p = (1 - self.decay_rate) ** k

            # Compute new candidates point according to the flipping probability
            flip_masks = np.array(
                np.random.choice([1, 0], p=[flipping_p, 1 - flipping_p], size=(self.n_samples, *flag.shape)),
                dtype=bool)
            w_candidates = np.logical_not(np.repeat(flag[None, :], self.n_samples, axis=0), where=flip_masks,
                                          out=np.repeat(flag[None, :], self.n_samples, axis=0).copy())

            # Evaluate all candidates
            evals_list = [self.cost_fn(w_val, **args) for w_val in
                          w_candidates]
            evals_cost = np.array([e[0] for e in evals_list])
            evals_x_approx = np.array([e[1] for e in evals_list])
            evals_new_rank = np.array([e[2] for e in evals_list])

            del evals_list

            # Search for descent direction
            ind = np.argmin(evals_cost)
            min_cost = evals_cost[ind]
            if min_cost < curr_cost:
                # Update tolerance
                if curr_cost - min_cost < self.tolerance:
                    tol -= 1
                else:
                    tol = self.patience
                # Update weights
                flag = w_candidates[ind]
                curr_cost = min_cost
                x_approx = evals_x_approx[ind]
                new_rank = evals_new_rank[ind]
            else:
                tol -= 1
            # Check patience stopping criterion
            if tol == 0:
                if self.verbose:
                    print(f"Reached patience stopping criterion at iteration {k}")
                break
            # Record weights, cost evaluation anf flipping prob
            weight_history.append(flag)
            cost_history.append([curr_cost, x_approx, new_rank])
            flipping_p_history.append(flipping_p)
        return weight_history, cost_history, flipping_p_history, k
