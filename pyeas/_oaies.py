import numpy as np
import math

from typing import Any
from typing import cast, Union
from typing import Optional  # telling the type checker that either an object of the specific type is required, or None is required
import time


class OAIES:
    """OpenAI-ES stochastic optimizer class with ask-and-tell interface.

    based off the style of: https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py


    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        optimiser:
            The optimisation method used: 'vaniller', 'momentum', 'adam'

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        cov:
            A covariance matrix (optional).
    """


    def __init__(
                self,
                alpha: float,
                sigma: float,
                bounds: np.ndarray,
                optimiser: str = 'adam',
                momentum: Optional[float] = None,
                population_size: Optional[Union[int, float]] = None,
                seed: Optional[int] = None,
                pop_dim_multiple: int = 0,
                groupings: Optional[Union[np.ndarray, list]] = None,
                mut_scheme: str = 'best1',
                constraint_handle: str = 'clip',
                ):
        

        # # Make random generator object
        self._rng = np.random.default_rng(seed)
        self._trial_seed = self._rng.integers(10000, size=1)[0]

        # # Check number of dimensions
        if groupings is None:
            self._n_dim = len(bounds)
        else:
            self._n_dim = np.sum(groupings)
        assert self._n_dim > 1, "The dimension of mean must be larger than 1"
        self._groupings = groupings
        self._bounds = np.array(bounds)

        # # Check population size
        assert pop_dim_multiple == 0 or pop_dim_multiple == 1, "population as multiple of number of dimensions flag must be 0 or 1"
        if population_size is None:
            self._popsize = 4 + math.floor(3 * math.log(self._n_dim))  # (eq. 48)  used for CMAES default allocation
        elif population_size is not None and pop_dim_multiple == 1:
            self._popsize = population_size*self._n_dim
        elif population_size is not None and pop_dim_multiple == 0:
            self._popsize = population_size
        assert self._popsize > 0, "popsize must be non-zero positive value."

        # # Check other hyper-params
        assert alpha > 0, "The value of mutation factor (i.e., F) must be larger than 0"
        assert isinstance(mut_scheme, str), "The mutation scheme (e.g., best1, rand1) must be a string"
        assert isinstance(constraint_handle, str), "The mutation boundary handle constrain (e.g., clip, reflection) must be a string"

        assert isinstance(optimiser, str), "The selected optimiser should be a string (e.g., vanilla, momentum, adam)"
        self._optimiser = optimiser
        
        if self._optimiser == 'momentum':
            self._m = momentum
            assert self._m > 0 and self._m < 1, "Momentum must be [0,1]"
        elif self._optimiser == 'adam':
            self._m = momentum
            assert self._m > 0 and self._m < 1, "Momentum must be [0,1]"

        self._alpha = alpha
        self._sigma = sigma

        self._prev_ga = None
        self._m = 0  # for adam GD
        self._v = 0  # for adam GD

        self._toggle = 0
        self._parent_norm = None
        self._parent_fit = None

        self.history = {}
        self.history['best_fits'] = []
        self.history['best_solutions'] = []

        return

    #    

    # #########################################
    # # Properties: https://www.freecodecamp.org/news/python-property-decorator/ 

    @property  # get 'protected' property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    @property
    def parent_pop(self) -> np.ndarray:
        """Return the denormalised (and grouped) parent poulation"""
        return self._denorm(self._pop_norm)

    @parent_pop.setter
    def parent_pop(self, new_parent_pop: np.ndarray):
        """Set the parent poulation"""
        self._pop_norm = self._norm(new_parent_pop)
        return

    @property
    def best_member(self) -> int:
        """Fetch the current best member and it's fitness"""
        fit = self._pop_fits[self._best_idx]
        member = self._denorm([self._pop_norm[self._best_idx]])[0]
        return (fit, member)
    
    #

    # #########################################
    # # Create Population members to evaluate

    def ask(self, loop: Optional[int] = None) -> np.ndarray:
        """Sample a whole trial population which needs to be evaluated"""

        assert self._toggle == 0, "Must first evaluate current trials and tell me their fitnesses."


        # # Generate population
        if self._pop_norm is None:
            self._pop_norm = self._sample_initi_pop() # generate initial parent population to evaluate
            self._toggle = 1
            return self._denorm(self._pop_norm)
        
        else:
            trial_pop = self._sample_trial_pop(loop)  # generate trial population to evaluate
            self._toggle = 1
            return self._denorm(trial_pop)
    

    def _sample_initi_pop(self) -> np.ndarray:
        """Sample initital normalised population"""
        norm_pop = []
        for i in range(self._popsize):
            norm_pop.append(np.around(self._rng.random(self._n_dim), decimals=5))
        return np.asarray(norm_pop)
    

    def _sample_trial_pop(self, loop) -> np.ndarray:
        """Sample trial normalised population"""

        # # Create number generator for trial member (optionally include loop to allow repetability)
        if loop is None:
            trial_rng = np.random.default_rng()
        else:
            trial_rng = np.random.default_rng(self._trial_seed+loop)

        # # Gen Gausian Pertubations To create trial psudo-population
        N = trial_rng.normal(size=(self._popsize, self._n_dim))

        trial_list = []
        for j in range(self._popsize):

            # cretae trial by adding noise
            trial = self._parent_norm + self._sigma*N[j]

            # perform boundary check
            checked_trial = self._mutant_boundary(trial)

            trial_list.append(checked_trial)
        
        trial_pop = np.asarray(trial_list, dtype=object)
        trial_pop = np.around(trial_pop.astype(np.float), decimals=5)

        return trial_pop

    #

    # # If a mutants value falls outide of the bounds, sort it out somehow

    def _mutant_boundary(self, mutant):
        """
        Ensures that a trial/mutant pop member does not violate given boundaries
        """
        reinit = 1
        resample_count = 0
        while reinit == 1:

            # If the mutants values violate the bounds, deal with it
            checked_mutant, reinit = self._handle_bound_violation(mutant)

            resample_count += 1

            if resample_count >= 100:
                checked_mutant, reinit = self._handle_bound_violation(mutant, force_select='clip')

        return checked_mutant

    def _handle_bound_violation(self, mutant, force_select=0):

        num_violations = self._count_bound_violation(mutant)

        # # if no violations, just return
        if num_violations == 0:
            return mutant, 0

        # # Implement the selected violation handeling sheme
        if force_select == 0:
            handle = self._constraint_handle
        else:
            handle = force_select

        #

        # # No handling
        if handle is None:
            return mutant, 0
        
        # # Perform projection (i.e., clipping)
        elif handle == 'clip' or handle == 'projection':
            mutant = np.clip(mutant, 0, 1)
            return mutant, 0

        # # Return the resample flag
        elif handle == 'resample':
            return mutant, 1

        # # Perform Scaled Mutant operation
        elif handle == 'scaled':
            alphas = [1]
            for m in mutant:
                if m > 1:
                    alphas.append(1/m)

            mutant = mutant*np.min(alphas)

            # # Not fool proof
            mutant = np.clip(mutant, 0, 1)

            return mutant, 0

        # # Perform Scaled Mutant operation
        elif handle == 'reflection':
            for i, m in enumerate(mutant):
                if m > 1:
                    mutant[i] = 2-m
                elif m < 0:
                    mutant[i] = -m
                else:
                    mutant[i] = m

            return mutant, 0
        
        else:
            raise ValueError("The constraint_handle that selects how to manage boundary violations is not valid")

    def _count_bound_violation(self, mutant):
        num_violations = np.size(np.where(mutant < 0)) + np.size(np.where(mutant > 1))  # How many clips are there?
        # print("Num clips below 0: %d, Num clips above 1: %d" % (np.size(np.where(mutant < 0)), np.size(np.where(mutant > 1))))
        return num_violations

    #

    # #########################################
    # # Use the fed back fitnesses to perform a generational update

    def tell(self, fitnessess: list, trials: Optional[np.ndarray] = None) -> None:
        """Tell the object the fitness values of the whole trial pseudo-population which has been valuated"""

        # # if condition returns False, AssertionError is raised:
        assert len(fitnessess) == self._popsize, "Must tell with popsize-length solutions."
        assert self._toggle == 1, "Must first ask (i.e., fetch) & evaluate new trials."


        # # Retrieve fitness information and make gradient decent update
        if self._pop_fits is None:
            self._parent_norm = trials[np.argmin(fitnessess)]
            self._parent_fit = np.min(fitnessess) 
            
        else:
            # evaluate trial population to evaluate
            # # Colapse the pseudo-population to update the parent/target
            assert trials is not None, "To perfom GD, please tell me the fitnesses and trials/pseudo-population used"
            trials_norm = self._norm(trials)

            R = -np.array(fitnessess)
            if np.std(R) <= 0:
                std = 1e-8
            else:
                std = np.std(R)
            A = (R - np.mean(R)) / std

            # # Grad Estimate
            g = 1/(self._popsize*self._sigma) * np.dot(trials_norm.T, A)

            # # Normal Grad Decent
            if self._optimiser == 'vanilla':
                ga = g*self._alpha
                parent_raw = parent_raw + ga

            # # Momentum
            elif self._optimiser == 'momentum':
                if self.prev_ga is None:
                    ga = g*self.prm['algo']['alpha']
                    parent_raw = parent_raw + ga
                else:
                    # https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
                    ga = g*self.prm['algo']['alpha'] + self.prm['algo']['momentum']*self.prev_ga
                    parent_raw = parent_raw + ga

            # # Adam GD
            elif self.prm['algo']['adam'] != 0:
                # https://machinelearningmastery.com/adam-optimization-from-scratch/
                # https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
                self.m = self.prm['algo']['adam_beta1']*self.m + (1-self.prm['algo']['adam_beta1'])*g
                self.v = self.prm['algo']['adam_beta2']*self.v + (1-self.prm['algo']['adam_beta2'])*g**2

                m_hat = self.m/(1-self.prm['algo']['adam_beta1']**t)
                v_hat = self.v/(1-self.prm['algo']['adam_beta2']**t)

                ga = self.prm['algo']['alpha']*m_hat/(1e-8+v_hat**0.5)
                parent_raw = parent_raw + ga

            else:
                raise ValueError("Invalid grad decent method")


        self.history['best_fits'].append(self.best_member[0])
        self.history['best_solutions'].append(self.best_member[1])

        self._toggle = 0
        return
    
    #

    # #########################################
    # # Functiosn which format the population

    def _denorm(self, norm_pop_in: np.ndarray) -> np.ndarray:
        """
        Produce the de-normalised population using the bounds.
        Also groups the population into its gene groups (if 'groupings' is being used).
        """

        if self._groupings is None:
            members_denorm = self._bounds[:,0] + norm_pop_in*abs(self._bounds[:,1]-self._bounds[:,0])
            pop_denorm = np.around(members_denorm.astype(np.float), decimals=5)
        else:
            pop_denorm = []
            for i, member in enumerate(norm_pop_in):
                pop_denorm.append(self._group_denorm(member))
        
        return np.asarray(pop_denorm, dtype=object)

    def _group_denorm(self, arr: np.ndarray) -> np.ndarray:

        grouped_arr = []
        st = 0
        for j, size in enumerate(self._groupings):
            lower_b, upper_b = self._bounds[j]  # get group bounds
            group = arr[st:(st+size)]
            group_denorm = lower_b + group*(abs(upper_b-lower_b))
            grouped_arr.append(np.around(group_denorm.astype(np.float), decimals=5))
            st += size

        return np.asarray(grouped_arr, dtype=object) 
    #

    def _norm(self, pop_in: np.ndarray) -> np.ndarray:
        """
        Produce the normalised population using the bounds.
        Also un-groups the population (if 'groupings' is being used).
        """

        if self._groupings is None:
            pop_norm = (pop_in - self._bounds[:,0])/(self._bounds[:,1] - self._bounds[:,0])
        else:
            pop_norm = np.array(pop_in)
            for j, grouping in enumerate(self._groupings):
                lower_b, upper_b = self._bounds[j]  # get group bounds
                pop_norm[:,j] = (pop_norm[:,j] - lower_b)/(upper_b - lower_b)
            pop_norm = np.array([np.concatenate(x) for x in pop_norm])

        pop_norm = np.around(pop_norm.astype(np.float), decimals=5)

        return np.asarray(pop_norm, dtype=object)

    def _ungroup_norm(self, grouped_arr: np.ndarray) -> np.ndarray:

        norm_member = []
        for j, group in enumerate(grouped_arr):
            lower_b, upper_b = self._bounds[j]  # get group bounds
            group_norm = (group - lower_b)/(upper_b - lower_b)
            norm_member.append(np.around(group_norm.astype(np.float), decimals=5))

        return np.concatenate(norm_member, axis=0)


    # #########################################
    # # Misc functions

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)
        return
    
    #

    # # fin
    
    