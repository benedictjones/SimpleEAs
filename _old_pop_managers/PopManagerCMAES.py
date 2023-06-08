# Import
import numpy as np
import time
import random


class population(object):
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


        return

    #

    def get_real_pop(self, pop): # gen_parent
        """
        Generate population in a "grouped by gene type" format
        """

        real_member = self._calc_denorm_(pop)

        return real_member

    #

    def get_norm_pop(self, real_pop): # gen_parent
        """
        Generate population in a "grouped by gene type" format
        """

        pop = self._calc_norm_(real_pop)

        return pop

    #

    def set_best(self, pop_real, pfits):

        bidx = np.argmin(pfits)
        self.parent = pop_real[bidx]

        return bidx

    #

    def set_mean(self, pop_real_mean):
        self.distibution_mean = self.get_real_pop([pop_real_mean])[0]
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
            """member = []
            for gen_group in self.prm['algo']['grouping']:
                member.append(np.around(self.rng.random(gen_group), decimals=4))"""
            member = np.around(self.rng.random(self.prm['algo']['dimensions']), decimals=4)
            pop.append(np.asarray(member, dtype=object))

        pop_raw = np.asarray(pop, dtype=object)
        pop = self._calc_denorm_(pop_raw)

        return pop_raw, pop

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

            # # Apply boundaries to updated parent
            member = self._format_mutant_(member)

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
