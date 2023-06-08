# Import
import numpy as np
import time
import random


class populationNES(object):
    """
    Generates and manges a population
    """

    def __init__(self, prm, seed=None):
        """
        Initialisation of class
        """

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
            self.trial_seed = self.rng.integers(10000, size=1)[0]

        # # assign dictionary to self
        self.prm = prm
        self.pcont = 0

        # # Set the previous ga to nothing, when updated it is used for momentum
        self.prev_ga = None
        self.m = 0  # for adam GD
        self.v = 0  # for adam GD
        self.alpha = self.prm['algo']['alpha'] # variable learning rate

        if prm['algo']['popsize'] <=1:
            raise ValueError("Need more than one sudo pop member!")

        # # Generate initial pop
        self.gen_parent()

        return

    #

    def gen_parent(self): # gen_parent
        """
        Generate population in a "grouped by gene type" format
        """
        tic = time.time()

        member = []
        for gen_group in self.prm['algo']['grouping']:
            member.append(np.around(self.rng.random(gen_group), decimals=5))
        parent = np.asarray(member, dtype=object)

        pop_out = np.asarray([parent], dtype=object)

        # self.parent_raw = pop_out[0]
        # print("\nTime to gen RAW initial pop:", time.time()-tic)

        self.parent = self._calc_denorm_(pop_out)[0]
        # print("\nTime to gen initial pop:", time.time()-tic)

        return

    #

    def gen_pop(self):
        """
        Generate population in a "grouped by gene type" format
        """
        tic = time.time()

        pop = []
        initi_popsize = int(self.prm['algo']['popsize']*self.prm['algo']['initi_pop_mult'])
        for i in range(initi_popsize):
            member = []
            for gen_group in self.prm['algo']['grouping']:
                member.append(np.around(self.rng.random(gen_group), decimals=5))
            pop.append(np.asarray(member, dtype=object))

        pop_raw = np.asarray(pop, dtype=object)
        pop = self._calc_denorm_(pop_raw)

        return pop_raw, pop

    #

    def set_parent(self, pop, pfits):

        bidx = np.argmin(pfits)
        #self.parent = pop[bidx]  # updated
        # print(bidx, ">set>", pop_raw[bidx][-1], pop[bidx][-1])

        parent_raw = self._calc_norm_([pop[bidx]])[0]
        parent_raw = self._mutate_boundary_(np.concatenate(parent_raw))
        parent_raw = self._format_mutant_(parent_raw)
        self.parent = self._calc_denorm_([parent_raw])[0]



        # self.parent_raw = self._calc_norm_([self.parent])[0]
        # self.parent = self._calc_denorm_([self.parent_raw])[0]

        # print(bidx, ">set updated>", self.parent_raw[-1], self.parent[-1])

        return bidx

    #

    def gen_trial_pop(self, loop_seed=None):
        """
        Generate a new trial population (both raw and denormalised).
        """
        tic = time.time()
        # print("\nloop_seed", loop_seed)

        # # Create all trial_mutants in a list
        trial_denorm_list = []
        trial_list = []
        parent_raw = self._calc_norm_([self.parent])[0]

        if loop_seed is None:
            trial_rng = np.random.default_rng()
        else:
            trial_rng = np.random.default_rng(self.trial_seed+loop_seed)

        # # Gen Gausian Pertubations To create trial psudo-population
        N = trial_rng.normal(size=(self.prm['algo']['popsize'], self.prm['algo']['dimensions']))

        # # Create Trial Pop
        for j in range(self.prm['algo']['popsize']):

            trial = np.concatenate(parent_raw) + self.prm['algo']['sigma']*N[j]
            trial = self._mutate_boundary_(trial)

            trial = self._format_mutant_(trial)
            trial_list.append(trial)

            trial_denorm = self._calc_denorm_([trial])[0]
            trial_denorm_list.append(trial_denorm)

        # print(trial_list[0])
        # exit()
        self.trial_pop_raw = np.asarray(trial_list, dtype=object)
        self.trial_pop = np.asarray(trial_denorm_list, dtype=object)
        # print("\nTime to gen trial pop:", time.time()-tic, " ")
        #exit()
        return

    #

    def refresh_parent(self, parent):
        """
        Used to refesh the parent member when training as an ELM.
        ELM training will alter the members output layer, this needs to be fed
        back to the PopManager object.
        """
        self.parent = parent
        # self.parent_raw = self._calc_norm_([parent])[0]
        return

    def refresh_trial_pop(self, trial_pop):
        """
        Used to refesh the trial pop when training as an ELM.
        ELM training will alter the members output layer, this needs to be fed
        back to the PopManager object.
        """
        self.trial_pop = trial_pop
        self.trial_pop_raw = self._calc_norm_(trial_pop)
        return

    #

    def update_pop(self, trial_fits, t=1):
        """
        Using the passed in fitnesses, the parent pop is updated, replacing
        members with those from the trial pop is a higher fitness was achived.
        """

        #Rdiff = (parent_fit-np.array(trial_fits))*10
        R = -np.array(trial_fits)
        '''
        #print("parent_fit:", parent_fit)
        print("trial_fits:", trial_fits)
        print("R:", R)
        #print(">>", parent_fit-trial_fits[0])
        #exit()
        #'''

        # # Update Parent raw values
        parent_raw = self._calc_norm_([self.parent])[0]

        if np.std(R) <= 0:
            std = 1e-8
        else:
            std = np.std(R)

        A = (R - np.mean(R)) / std
        #print("Rdiff", Rdiff[:4])
        #print("R", R[:4])
        #print("A", A[:4])
        #A = Rdiff
        #exit()

        # # Grad Estimate
        g = 1/(self.prm['algo']['popsize']*self.prm['algo']['sigma']) * np.dot(self.trial_pop_raw.T, A)

        """if self.prm['algo']['ELM'] != 0:
            g[-2] = g[-2]*-1
            g[-1] = g[-1]*-1"""

        #print("\ng:\n", g[-2])
        #print("\n p raw:\n", self.parent_raw)
        #print("p concact:", np.concatenate(self.parent_raw))

        # # Normal Grad Decent
        if self.prm['algo']['momentum'] == 0 and self.prm['algo']['adam'] == 0:
            ga = g*self.prm['algo']['alpha']
            parent_raw = parent_raw + ga

        # # Momentum
        elif self.prm['algo']['momentum'] > 0 and self.prm['algo']['momentum'] < 1:
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

        # # Save prev (for momentum calc)
        #print("\nga:\n", ga[-2])
        #exit()
        self.prev_ga = ga


        # # Apply boundaries to updated parent
        parent_raw = self._mutate_boundary_(np.concatenate(parent_raw))
        parent_raw = self._format_mutant_(parent_raw)

        # # Get Parent
        self.parent = self._calc_denorm_([parent_raw])[0]

        return

    #

    #

    #

    #

    '''
    ###############################################################
    If a mutants value falls outide of the bounds, sort it out somehow
    ###############################################################
    '''

    def _mutate_boundary_(self, mutant):
        """
        Selects which mutation scheme to use, and returns the mutant.
        """
        reinit = 1
        resample_count = 0
        og_mutant = mutant.copy()
        while reinit == 1:

            # If the mutants values violate the bounds, deal with it
            handled_mutant, reinit = self.handle_bound_violation(mutant)

            resample_count += 1

            if resample_count >= 100:
                handled_mutant, reinit = self.handle_bound_violation(mutant, force_select='reflection')
                # print("\n --- Resample limit hit!! Num violations:", self.count_bound_violation(mutant))

        #print("Post handle, num violations:", self.count_bound_violation(mutant), " , num re-samples:", resample_count)

        # # Don't apply boundary to ELM output layer
        # # Assign Original output weight and bias if ELM trained
        if self.prm['algo']['ELM'] == 1:
            handled_mutant[-2] = og_mutant[-2]
            handled_mutant[-1] = og_mutant[-1]
        elif self.prm['algo']['ELM'] == 'initi' and self.pcont == 0:
            print("\nIniti ELM trainined pop boundary is handled: %s" % (self.prm['algo']['constraint_handle']))
            self.pcont = 1

        return handled_mutant

    #
    #

    def handle_bound_violation(self, mutant, force_select=0):

        num_violations = self.count_bound_violation(mutant)

        # # if no violations, just return
        if num_violations == 0:
            return mutant, 0

        #

        # # Implement the selected violation handeling sheme
        if force_select == 0:
            handle = self.prm['algo']['constraint_handle']
        else:
            handle = force_select

        #

        # # Perform projection (i.e., clipping)
        if handle == 'clip' or handle == 'projection':
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

            """
            if self.count_bound_violation(mutant) != 0:
                print("alphas:", alphas)
                print("Num violations:", self.count_bound_violation(mutant))
                print(mutant)
                raise ValueError("Scaled Mutant method did not work!")
            # """

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

    #

    #

    def count_bound_violation(self, mutant):

        num_violations = np.size(np.where(mutant < 0)) + np.size(np.where(mutant > 1))  # How many clips are there?
        # print(" > Total bad clips:", num_violations , " out of:", self.num_flexy_genes)
        # print("Num clips below 0: %d, Num clips above 1: %d" % (np.size(np.where(mutant < 0)), np.size(np.where(mutant > 1))))

        return num_violations

    #


    #

    #

    #

    #

    #

    #

    #

    '''
    ###############################################################
    Population and mutant generation function
    ###############################################################
    '''

    def _format_mutant_(self, mutant):
        """
        Re-format mutant into "grouped by gene type" format
        Note: Don't make arrays dtype=object, as they they can't be rounded.
        """
        member = []
        i = 0
        for group_dim in self.prm['algo']['grouping']:
            gen_group = []
            for g in range(group_dim):
                gen_group.append(mutant[i])
                i = i + 1
            member.append(np.asarray(gen_group))
            # member.append(np.asarray(gen_group, dtype=object))

        return np.asarray(member, dtype=object)

    #

    def _calc_denorm_(self, pop_in):
        """
        Produce the de-normalised population using the bounds.
        Also groups the population into its gene groups, which are indexed
        using the 'loc' variable.

        """
        # print("\n_calc_denorm_")
        # pop_denorm = np.around(self.min_b + pop_in * self.diff, decimals=3)  # pop with their real values

        # # Group and scale the Population
        pop_denorm = []
        for i, member in enumerate(pop_in):
            member_denorm = []

            for j, genome_group in enumerate(member):
                lower_b, upper_b = self.prm['algo']['bounds'][j]  # get group bounds
                gen_group_denorm = lower_b + genome_group*(upper_b-lower_b)
                gen_group_denorm = np.around(gen_group_denorm.astype(np.double), decimals=5)
                member_denorm.append(np.asarray(gen_group_denorm))
            pop_denorm.append(np.asarray(member_denorm, dtype=object))

        pop_denorm = np.asarray(pop_denorm, dtype=object)

        return np.asarray(pop_denorm, dtype=object)

    #

    def _calc_norm_(self, denorm_pop_in):
        """
        Produce the normalised population using the bounds.
        Also groups the population into its gene groups, which are indexed
        using the 'loc' variable.

        """

        # # Group and scale the Population
        norm_pop = []
        for i, member_denorm in enumerate(denorm_pop_in):
            member_norm = []

            for j, genome_group in enumerate(member_denorm):
                lower_b, upper_b = self.prm['algo']['bounds'][j]  # get group bounds
                gen_group_norm = (genome_group - lower_b)/(upper_b-lower_b)
                gen_group_norm = np.around(gen_group_norm.astype(np.double), decimals=5)

                member_norm.append(np.asarray(gen_group_norm))

            norm_pop.append(np.asarray(member_norm, dtype=object))

        norm_pop = np.asarray(norm_pop, dtype=object)

        return norm_pop
    #

    #



#

#

# fin
