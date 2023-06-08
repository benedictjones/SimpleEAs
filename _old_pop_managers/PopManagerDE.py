# Import
import numpy as np
import time
import random


class populationDE(object):
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
        self.best_idx = 'na'

        # # Generate initial pop
        self.gen_pop()

        return

    #

    def gen_pop(self):
        """
        Generate population in a "grouped by gene type" format
        """
        tic = time.time()

        pop = []
        for i in range(self.prm['algo']['popsize']):
            member = []
            for gen_group in self.prm['algo']['grouping']:
                member.append(np.around(self.rng.random(gen_group), decimals=5))
            pop.append(np.asarray(member, dtype=object))

        pop_out = np.asarray(pop, dtype=object)

        self.pop_raw = pop_out
        # print("\nTime to gen RAW initial pop:", time.time()-tic)

        self.pop = self._calc_denorm_(pop_out)
        # print("\nTime to gen initial pop:", time.time()-tic)

        return

    #

    def gen_trial_pop(self, loop=None):
        """
        Generate a new trial population (both raw and denormalised).
        """
        tic = time.time()
        # # Create all trial_mutants in a list
        trial_denorm_list = []
        trial_list = []
        for j in range(self.prm['algo']['popsize']):

            if loop is None:
                self.trial_rng = np.random.default_rng()
            else:
                self.trial_rng = np.random.default_rng(self.trial_seed+loop+j)

            # # Select Indexs to generate mutants from
            """
            Creates an range array(0, popsize) but excludes the current
            value of j, used to randomly select pop involved in mutation.
            i.e idxs is all pop index's except the current one
            """
            idxs = [idx for idx in range(self.prm['algo']['popsize']) if idx != j]

            # # Mutation
            mutant = self._mutate_(idxs, j)

            # # Recombination & Replacement
            trial_denorm, trial = self._bin_cross_(j, self.pop_raw, mutant, loop)
            # trial_denorm, trial = self._bin_cross_WCST_(j, pop, mutant, i)

            #print(trial)
            #exit()

            trial_denorm_list.append(trial_denorm)
            trial_list.append(trial)

        self.trial_pop_raw = np.asarray(trial_list, dtype=object)
        self.trial_pop = np.asarray(trial_denorm_list, dtype=object)
        # print("\nTime to gen trial pop:", time.time()-tic, " ")
        #exit()
        return

    #

    def update_pop(self, parent_fits='na', trial_fits='na'):
        """
        Using the passed in fitnesses, the parent pop is updated, replacing
        members with those from the trial pop is a higher fitness was achived.
        """

        if (isinstance(trial_fits, str) is True and trial_fits == 'na') and isinstance(parent_fits, str) is False:
            self.pop_fits = np.array(parent_fits)
            self.best_idx = np.argmin(self.pop_fits)  # find best Training idex
            self.best_fit = self.pop_fits[self.best_idx]  # use independant best fit

        elif (isinstance(parent_fits, str) is True and parent_fits == 'na') and isinstance(trial_fits, str) is False:
            # # Compare the children/trial genomes to the parent/target
            for j in range(self.prm['algo']['popsize']):
                # print("compare fi", j, "fi=", fi[j], "to previous fitness=", fitness[j])
                # print("     For Genome:", str(np.around(trial_denorm_array[j], decimals=3)))

                if trial_fits[j] <= self.pop_fits[self.best_idx]:  # assign best genome/pop
                    self.best_idx = j  # find best Training idex

                if trial_fits[j] <= self.pop_fits[j]:  # whether to keep parent or child/trial
                    self.pop_raw[j] = self.trial_pop_raw[j]  # update corresponding pop
                    self.pop_fits[j] = trial_fits[j]

            # # Update Best fit
            self.best_fit = self.pop_fits[self.best_idx]

            # # Update denorm pop using the newly updated pop
            self.pop = self._calc_denorm_(self.pop_raw)

        else:
            raise ValueError("Onle pass trial or parent fits into update_pop.")

        return

    #

    def refresh_parent(self, parent_pop):
        """
        Used to refesh the parent member when training as an ELM.
        ELM training will alter the members output layer, this needs to be fed
        back to the PopManager object.
        """
        self.pop = parent_pop
        self.pop_raw = self._calc_norm_(parent_pop)
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

    #

    #

    '''
    ###############################################################
    Mutation Functions
    ###############################################################
    '''

    def _mutate_(self, idxs, current_idx):
        """
        Selects which mutation scheme to use, and returns the mutant.
        """
        reinit = 1
        resample_count = 0
        while reinit == 1:

            if self.prm['algo']['mut_scheme'] == 'rand1':
                mutant = self._rand1_(idxs, self.pop_raw)

            elif self.prm['algo']['mut_scheme'] == 'best1':
                mutant = self._best1_(idxs, self.pop_raw)

            elif self.prm['algo']['mut_scheme'] == 'rand2':
                mutant = self._rand2_(idxs, self.pop_raw)

            elif self.prm['algo']['mut_scheme'] == 'best2':
                mutant = self._best2_(idxs, self.pop_raw)

            elif self.prm['algo']['mut_scheme'] == 'ttb1':
                mutant = self._ttb1_(idxs, self.pop_raw, current_idx)

            else:
                raise ValueError("Invalit Mutation Scheme: %s" % (self.prm['algo']['mut_scheme']))

            # If the mutants values violate the bounds, deal with it
            mutant, reinit = self.handle_bound_violation(mutant)

            resample_count += 1

            if resample_count >= 100:
                mutant, reinit = self.handle_bound_violation(mutant, force_select='reflection')
                # print("\n --- Resample limit hit!! Num violations:", self.count_bound_violation(mutant))

        #print("Post handle, num violations:", self.count_bound_violation(mutant), " , num re-samples:", resample_count)

        return mutant

    #
    #

    def _rand1_(self, idxs, pop):
        """
        Random1 mutation method.
        Randomly choose 3 indexes without replacement.
        """
        # selected = np.random.choice(idxs, 3, replace=False)
        selected = self.trial_rng.choice(idxs, 3, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        a, b, c = np_pop[selected]  # assign to a variable
        # note this is not the real pop values

        # mutant
        mutant = np.concatenate(a) + self.prm['algo']['mut'] * (np.concatenate(b) - np.concatenate(c))

        return mutant  # this is unformatted

    #

    def _rand2_(self, idxs, pop):
        """
        Random2 mutation method.
        Randomly choose 5 indexes without replacement
        """

        # selected = np.random.choice(idxs, 5, replace=False)
        selected = self.trial_rng.choice(idxs, 5, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        a, b, c, d, e = np_pop[selected]  # assign to a variable
        # note; a, b etc are genomes

        # mutant
        mutant = np.concatenate(a) + self.prm['algo']['mut'] * (np.concatenate(b) - np.concatenate(c) + np.concatenate(d) - np.concatenate(e))

        return mutant  # this is unformatted

    #

    def _best1_(self, idxs, pop):
        """
        Best1 mutation method
        Randomly choose 2 indexes without replacement, combined with the best.
        """

        # selected = np.random.choice(idxs, 2, replace=False)
        selected = self.trial_rng.choice(idxs, 2, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        b, c = np_pop[selected]
        a = pop[self.best_idx]

        # mutant
        mutant = np.concatenate(a) + self.prm['algo']['mut'] * (np.concatenate(b) - np.concatenate(c))

        #print("mutated:", mutant)

        return mutant  # this is unformatted

    #

    def _best2_(self, idxs, pop):
        """
        Best2 mutation method
        Randomly choose 4 indexes without replacement, combined with the best.
        """

        # selected = np.random.choice(idxs, 4, replace=False)
        selected = self.trial_rng.choice(idxs, 4, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        b, c, d, e = np_pop[selected]
        a = pop[self.best_idx]

        # mutant
        mutant = np.concatenate(a) + self.prm['algo']['mut'] * (np.concatenate(b) - np.concatenate(c) + np.concatenate(d) - np.concatenate(e))

        return mutant  # this is unformatted

    #

    def _ttb1_(self, idxs, pop, current_idx):
        """
        Target-to-best/1 mutation method
        Randomly choose 4 indexes without replacement, combined with the best.
        """

        # selected = np.random.choice(idxs, 2, replace=False)
        selected = self.trial_rng.choice(idxs, 2, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        a = pop[current_idx]
        b = pop[self.best_idx]
        c, d = np_pop[selected]


        F1 = self.prm['algo']['mut']
        F2 = self.prm['algo']['mut']

        # mutant
        mutant = np.concatenate(a) + F1 * (np.concatenate(b) - np.concatenate(a)) + F2 * (np.concatenate(c) - np.concatenate(d))

        return mutant  # this is unformatted

    #

    '''
    ###############################################################
    If a mutants value falls outide of the bounds, sort it out somehow
    ###############################################################
    '''

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

    '''
    ###############################################################
    Recombination & Replacement
    ###############################################################
    '''

    def _bin_cross_(self, j, pop, mutant, epoch):
        """
        Basic binary crossover
        """

        # # Apply crossp model
        if self.prm['algo']['crossp_scheme'] == 'none':
            cr = self.prm['algo']['crossp']

        elif self.prm['algo']['crossp_scheme'] == 'linear':
            cr = 1 - epoch*(1/self.prm['algo']['epochs'])

        elif self.prm['algo']['crossp_scheme'] == 'quad':
            cr = 1 - (epoch*(1/self.prm['algo']['epochs']))**2


        # # Return true or false for each of the random elements
        # cross_points = np.random.rand(self.prm['algo']['dimensions']) < cr
        cross_points = self.trial_rng.random(size=self.prm['algo']['dimensions']) < cr

        # # Randomly set a paramater to True to ensure a mutation occurs
        a = np.arange(self.prm['algo']['dimensions'])
        x = int(self.trial_rng.choice(a))

        cross_points[x] = True

        # # Where True, yield x, otherwise yield y np.where(condition,x,y)
        trial = np.where(cross_points, mutant, np.concatenate(pop[j]))

        # # put into the grouped by gene type format used
        trial = self._format_mutant_(trial)

        trial_denorm = self._calc_denorm_([trial])
        #trial_denorm = self.min_b + trial * self.diff  # terms of the real val
        # will this be a problem with binary number/pin posistions

        return trial_denorm[0], trial

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
                gen_group_denorm = lower_b + genome_group*(abs(upper_b-lower_b))
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
