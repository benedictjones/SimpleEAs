import numpy as np
import math

from typing import Any
from typing import cast, Union
from typing import Optional  # telling the type checker that either an object of the specific type is required, or None is required
import time

from pyeas._population_scaling import PopScale


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
            The optimisation method used: 'vanilla', 'momentum', 'adam'

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

        pop_dim_multiple:
            Boolean toggle to set the population as a multiple of the number of dimesnions (optional)

        cov:
            A covariance matrix (optional).
    """


    def __init__(
                self,
                alpha: float,
                sigma: float,
                bounds: np.ndarray,
                optimiser: str = 'adam',
                population_size: Optional[Union[int, float]] = None,
                seed: Optional[int] = None,
                pop_dim_multiple: int = 0,
                groupings: Optional[Union[np.ndarray, list]] = None,
                constraint_handle: Optional[str] = None,
                momentum: Optional[float] = None,
                beta1=0.9,
                beta2=0.99,
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

        # # Initialise object to normalise and denormalise the population
        self.PopScale = PopScale(np.array(bounds), groupings)

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
        assert alpha > 0, "The value of alpha (i.e., learning rate) must be larger than 0"
        assert sigma > 0, "The value of sigma (i.e., the pseudo-population noise) must be larger than 0"

        self._alpha = alpha
        self._sigma = sigma

        assert isinstance(optimiser, str), "The selected optimiser should be a string (e.g., vanilla, momentum, adam)"
        self._optimiser = optimiser
        self._prev_ga = None

        if self._optimiser == 'momentum':
            assert momentum > 0 and momentum < 1, "Momentum must be [0,1]"
            self._m = momentum
        elif self._optimiser == 'adam':
            self._m = 0
            self._v = 0
            assert beta1 > 0, "The value of adam's beta1 must be larger than 0"
            assert beta2 > 0, "The value of adam's beta2 must be larger than 0"
            self._beta1 = beta1
            self._beta2 = beta2

    
        self._toggle = 0
        self._toggle_parent = 0
        self._parent_norm = None
        self._parent_fit = None
        self._constraint_handle = constraint_handle
        self._number_evals = 0  # number of training evaluations

        self.history = {}
        self.history['best_fits'] = []
        self.history['best_solutions'] = []
        self.history['num_evals'] = []

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
    def bounds(self) -> int:
        """The bounds"""
        return self.PopScale._bounds

    @property  
    def groupings(self) -> int:
        """The Grouping"""
        return self.PopScale._groupings

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    @property
    def parent(self) -> np.ndarray:
        """Return the denormalised (and grouped) parent poulation"""
        return self.PopScale._denorm([self._parent_norm])[0]

    @parent.setter
    def parent(self, new_parent_pop: np.ndarray):
        """Set the parent poulation"""
        self._parent_norm = self.PopScale._norm([new_parent_pop])[0]
        return

    @property
    def best_trial(self) -> np.ndarray:
        """Fetch the current psudo-population best member and it's fitness"""
        fit = np.min(self._trial_fits)
        member = self.PopScale._denorm([self._trials[np.argmin(self._trial_fits)]])[0]
        return (fit, member)
    
    @property
    def best(self) -> np.ndarray:
        """Return the denormalised (and grouped) parent poulation"""
        return (self.history['best_fits'][-1], self.history['best_solutions'][-1])

    @property
    def evals(self) -> int:
        """The number of evaluations (i.e., number of computations)"""
        return self._number_evals
    

    #

    # #########################################
    # # Create Population members to evaluate

    def ask(self, loop: Optional[int] = None) -> np.ndarray:
        """Sample a whole trial population which needs to be evaluated"""

        assert self._toggle == 0, "Must first evaluate current trials and tell me their fitnesses."
        assert self._toggle_parent == 0, "Must first evaluate and set the best/parent member fitness"

        # # Generate population
        if self._parent_norm is None:
            self._parent_norm = self._sample_initi_pop() # generate initial parent population to evaluate
            self._toggle = 1
            return self.PopScale._denorm([self._parent_norm])[0]
        
        else:
            trial_pop = self._sample_trial_pop(loop)  # generate trial population to evaluate
            self._toggle = 1
            return self.PopScale._denorm(trial_pop)
    

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

    def tell(self, fitnessess: list, trials: Optional[np.ndarray] = None, t: int=1) -> None:
        """Tell the object the fitness values of the whole trial pseudo-population which has been valuated
            Args:

                fitnessess:
                    List of fitnesses for the corresponding trial members (i.e., pseudo-population)

                trials:
                    The trial members (i.e., pseudo-population) considered.

                t:
                    The iteration (optional). If not iterated, a static decay schedule is used.
        """

        # # if condition returns False, AssertionError is raised:
        assert len(fitnessess) == self._popsize, "Must tell with popsize-length solutions."
        assert self._toggle == 1, "Must first ask (i.e., fetch) & evaluate new trials."
        assert t >= 0, "The time/iteration must be greater than zero."
        assert self._toggle_parent == 0, "Must first evaluate and set the best/parent member fitness"

        self._trials = trials
        self._trial_fits = fitnessess

        # # Retrieve fitness information and make gradient decent update
        if self._parent_fit is None:
            # # Use a uniformly generated population to select a starting location
            self._parent_norm = self.PopScale._norm([trials[np.argmin(fitnessess)]])[0]
            self._parent_fit = np.min(fitnessess) 
        else:
            # # Colapse the pseudo-population to update the parent/target
            assert trials is not None, "To perfom GD, please tell me the fitnesses and trials/pseudo-population used"
            trials_norm = self.PopScale._norm(trials)
            #print("self._parent_norm", self._parent_norm)
            theta = np.copy(self._parent_norm)  # parent member to perfrom GD on

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
                theta = theta + ga

            # # Momentum
            elif self._optimiser == 'momentum':
                # https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
                if self._prev_ga is None:
                    ga = g*self._alpha
                    theta = theta + ga
                else:
                    ga = g*self._alpha + self._m*self._prev_ga
                    theta = theta + ga
                self._prev_ga = ga

            # # Adam GD
            elif self._optimiser == 'adam':
                # https://machinelearningmastery.com/adam-optimization-from-scratch/
                # https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
                self._m = self._beta1*self._m + (1-self._beta1)*g
                self._v = self._beta2*self._v + (1-self._beta2)*g**2

                m_hat = self._m/(1-self._beta1**t)
                v_hat = self._v/(1-self._beta2**t)

                ga = self._alpha*m_hat/(1e-8+v_hat**0.5)
                theta = theta + ga

            else:
                raise ValueError("Invalid gradient decent optimiser method")

            #print("theta:", theta)
            # print("ga:", np.around(ga.astype(np.float), decimals=5))
            theta = np.around(theta.astype(np.float), decimals=5)
            theta = self._mutant_boundary(theta)
            self._parent_norm = theta
            #print("theta:", theta)
            #exit()
        
        self._number_evals += len(trials)
        self.history['num_evals'].append(self._number_evals)

        self._toggle = 0
        self._toggle_parent = 1
        return
    
    def tellAgain(self, parent_fit: float):
        """Set the parent member's fitness"""

        assert self._parent_fit is not None, "Can't set the best fitness until you have evaluated generation zero"
        assert self._toggle == 0, "Can only set the best result once you have updated the pop via a tell."
        assert self._toggle_parent == 1, "Must first perform a generational update, then inform about the new parent fit (e.g., optimizer.best = parent_fit)"
        # assert parent_fit >= 0, "The parent fitness score must be greater than zero."

        self._parent_fit = parent_fit
        
        # # Don't increase number of evaluations, as this doesn't progress evo
        # self._number_evals += 1

        self.history['best_fits'].append(parent_fit)
        self.history['best_solutions'].append(self.parent)

        self._toggle_parent = 0
        return
        
    #

    # #########################################
    # # Misc functions

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)
        return
    
    #

    # # fin
    
    