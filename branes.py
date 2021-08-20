import numpy as np
from fastprogress import master_bar, progress_bar



def GA(nRuns, nGens, popSize, nSurvive, bix2, UI, maxStacks, filler, signfix,  # Meta/environment
       xoverProbs, mutRates,  # Cross-over/mutation
       weights, tadScale, susyScale,  # Fitness
       progress,  # Progress bar
       saveTKS, tksFolder, tksFileName,  # Saving models
       saveFill, fillFilePath,  # Saving filler distributions
       saveOpt, optFilePath  # Saving for optimization (TKS distributions)
       ):
    
    for n in range(nRuns):
        print('Run %2d' % (n+1))

        # Boolean for when first TKS model is found
        firstFound = False

        # Empty arrays for storing all (partially) consistent individuals found over the generations
        TKSstrings = ['T', 'K', 'S', 'TK', 'TS', 'KS', 'TKS']
        TKSmodels = [[np.empty([0, n, 7], dtype='int') for n in range(maxStacks + 1)]
                     for s in TKSstrings]


        # Initialize population
        pop = population(popSize, maxStacks)
        # Compute fitness for all new individuals of population
        for e in pop.individuals:
            e.computeFitness(bix2, UI, filler, weights, tadScale, susyScale)

        rewardHist = np.zeros([nGens+1, popSize, 4])
        rewardHist[0] = pop.getRewards()

        if progress:

            # Keep track of summary statistics
            fitnessQuantiles = np.zeros([nGens + 1, 4])
            fittestRewards = np.zeros([nGens + 1, 5])
            fillerAve = np.zeros([nGens + 1])
            nStacksAve = np.zeros([nGens + 1])
            TKScounts = np.zeros([nGens + 1, len(TKSstrings)])

            # Fill for initial population
            allFits = pop.getFitnesses()
            fitnessQuantiles[0] = np.quantile(allFits, [0.25, 0.50, 0.75, 1.00])
            fittestRewards[0] = pop.individuals[np.argsort(allFits)[-1]].fitnessDetails
            fillerAve[0] = pop.getFillerAve()
            nStacksAve[0] = pop.getNStacksAve() / maxStacks
            TKScounts[0] = [len(a) for a in TKSmodels]

            # Progress bar for monitoring fitness distribution over the generations
            progBar = master_bar(range(1, nGens + 1))
            progBar.names = ['', '', 'tad', 'K-th', 'SUSY', 'MSSM', 'fillers', 'max', 'Q75', 'Q50', 'Q25',
                             'fillerAve', 'nStacksAve', 'T', 'K', 'S', 'TK', 'TS', 'KS', 'TKS']
            
            # Progress bar serves as iterator
            gIter = progBar

        else:
            # Iterator
            gIter = range(1, nGens + 1)


        for g in gIter:

            # Breed next generation
            pop.breed(nSurvive, bix2, filler, signfix, xoverProbs, mutRates)

            if progress:
                # Progress bar serves as iterator
                iIter = progress_bar(range(popSize), parent=progBar)
            else:
                # Iterator
                iIter = range(popSize)

            # Compute fitness for all new individuals of population
            for i in iIter:
                pop.individuals[i].computeFitness(bix2, UI, filler, weights, tadScale, susyScale)

            # Get this generation's T/K/S individuals
            TKSnew = [pop.getConsistent(s) for s in TKSstrings]
            for i in range(len(TKSstrings)):
                for e in TKSnew[i]:
                    nStacks = len(e.stacks)
                    TKSmodels[i][nStacks] = np.append(TKSmodels[i][nStacks],
                                                      [saveSort(e.stacks)], axis=0)
                for n in range(maxStacks + 1):
                    if len(TKSmodels[i][n]) > 0:
                        TKSmodels[i][n] = np.unique(TKSmodels[i][n], axis=0)

            if len(TKSnew[-1]) > 0 and not firstFound:
                # Print when first TKS individual found
                firstFound = True
                print('First TKS : generation %d' % g)

            rewardHist[g] = pop.getRewards()

            if progress:
                # Update fitness summary statistics
                allFits = pop.getFitnesses()
                fitnessQuantiles[g] = np.quantile(allFits, [0.25, 0.50, 0.75, 1.00])
                fittestRewards[g] = pop.individuals[np.argsort(allFits)[-1]].fitnessDetails
                fillerAve[g] = pop.getFillerAve()
                nStacksAve[g] = pop.getNStacksAve() / maxStacks
                TKScounts[g] = [len(a) for a in TKSnew]

                # Update graph of fitness summary statistics
                progBar.update_graph([
                    [[0, g], [1, 1]],
                    [[1.05*g, 2.05*g], [1, 1]],
                    [np.arange(1.05*g, 2.05*g+1), fittestRewards[:g+1, 0]],
                    [np.arange(1.05*g, 2.05*g+1), fittestRewards[:g+1, 1]],
                    [np.arange(1.05*g, 2.05*g+1), fittestRewards[:g+1, 2]],
                    [np.arange(1.05*g, 2.05*g+1), fittestRewards[:g+1, 3]],
                    [np.arange(1.05*g, 2.05*g+1), fittestRewards[:g+1, 4]],
                    [np.arange(g+1), fitnessQuantiles[:g+1, 3]],
                    [np.arange(g+1), fitnessQuantiles[:g+1, 2]],
                    [np.arange(g+1), fitnessQuantiles[:g+1, 1]],
                    [np.arange(g+1), fitnessQuantiles[:g+1, 0]],
                    [np.arange(1.05*g, 2.05*g+1), fillerAve[:g+1]],
                    [np.arange(1.05*g, 2.05*g+1), nStacksAve[:g+1]],
                    [np.arange(g+1), TKScounts[:g+1, 0]/popSize],
                    [np.arange(g+1), TKScounts[:g+1, 1]/popSize],
                    [np.arange(g+1), TKScounts[:g+1, 2]/popSize],
                    [np.arange(g+1), TKScounts[:g+1, 3]/popSize],
                    [np.arange(g+1), TKScounts[:g+1, 4]/popSize],
                    [np.arange(g+1), TKScounts[:g+1, 5]/popSize],
                    [np.arange(g+1), TKScounts[:g+1, 6]/popSize]
                ], [-0.1*g, 2.5*g], [0, 1.05], figsize=(13, 5))


        # Print number of consistent solutions found
        print('\tn   | ', end='')
        for n in range(2, maxStacks + 1):
            print('%6d' % n, end='')
        print(' | total\n\t' + '-'*(8+6*maxStacks))
        for i in range(len(TKSstrings)):
            print('\t%-3s | ' % TKSstrings[i], end='')
            for n in range(2, maxStacks + 1):
                a = len(TKSmodels[i][n])
                if a > 0:
                    print('%6d' % a, end='')
                else:
                    print(' '*6, end='')
            a = sum([len(c) for c in TKSmodels[i]])
            if a > 0:
                print(' |%6d' % a)
            else:
                print(' |' + ' '*6)


        # if saveFill:
        #     try:
        #         # Load previously saved
        #         loaded = np.load(fillFilePath)
        #         toSave = np.append(loaded, [fillerAve * nStacks], axis=0)
        #     except IOError:
        #         # No such filed exists
        #         toSave = np.array([fillerAve * nStacks])
            
        #     # Save to file
        #     np.save(fillFilePath, toSave)
        
        if saveOpt:
            TKScounts = np.array([len(tksn) for tksn in TKSmodels[-1]])
            
            try:
                # Load previously saved
                loaded = np.load(optFilePath)
                toSave = loaded + TKScounts
            except IOError:
                # No such filed exists
                toSave = TKScounts
            
            # Save to file
            np.save(optFilePath, toSave)


        if saveTKS:
            for i in range(len(TKSstrings)):
                for n in range(maxStacks + 1):
                    if len(TKSmodels[i][n]) > 0:
                        filePath = tksFolder + TKSstrings[i] + ("/ns=%02d_" % n) + tksFileName
                        saveSolutions(TKSmodels[i][n], filePath)

        print('')
        pop.displayFittest(1, bix2)

        return rewardHist


class population:

    def __init__(self, size, maxStacks):
        self.size = size

        # Initialize array of random individuals
        self.individuals = np.array([individual(maxStacks) for i in range(size)])


    def breed(self, nSurvive, bix2, filler, signfix, xoverProbs, mutRates):

        # Create empty array for next generation and copy
        # over 'nSurvive' of the fittest individuals
        newIndividuals = np.empty([self.size], dtype='object')
        newIndividuals[:nSurvive] = self.getFittest(nSurvive)

        # Loop until next generation is full
        i = nSurvive

        while i < self.size:

            # Select two parents
            parent1 = self.binaryTournament()
            parent2 = self.binaryTournament()

            # Generate child by crossover
            child = parent1.crossOver(parent2, xoverProbs)

            # - Apply mutations
            # - Ensure Na>0 and coprime windings
            # - Correct signs of type A'/B'/C'
            # - Bring to standard form
            child.mutate(mutRates, bix2)
            child.makeValid(filler, bix2)
            if signfix:
                child.correctSigns(bix2)
            child.standardize()

            if len(child.stacks) >= 2:
                # Add child to next generation
                newIndividuals[i] = child
                i += 1

        # Update population to newly created generation
        self.individuals = newIndividuals


    def binaryTournament(self):
        # Select two random individuals and return the fittest
        i, j = np.random.randint(self.size, size=2)

        if self.individuals[i].fitness > self.individuals[j].fitness:
            return self.individuals[i]
        else:
            return self.individuals[j]


    def getFitnesses(self):
        # Returns list of the fitness for all individuals in the population
        return [e.fitness for e in self.individuals]


    def getFittest(self, n):
        # Returns the n fittest individuals in the population
        fits = self.getFitnesses()
        bestn = np.argsort(fits)[-1:-(n+1):-1]
        return self.individuals[bestn]

    def getRewards(self):
        return np.array([e.rewards for e in self.individuals])

    def getConsistent(self, s):
        # s should be a string: T, K, S, TK, TS, KS, or TKS
        # e.consCond for each individiual e is a string recording which of
        # the tadpole, K-theory and SUSY consistency conditions are satisfied
        return self.individuals[[e.consCond == s for e in self.individuals]]


    def getFillerAve(self):
        # Returns the average fraction of stacks which are filler branes
        return np.mean([e.fitnessDetails[4] for e in self.individuals])


    def getNStacksAve(self):
        # Returns the average number of stacks
        return np.mean([len(e.stacks) for e in self.individuals])


    def displayFittest(self, n, bix2):
        # Display the n fittest individuals in the population
        for e in self.getFittest(n):
            e.display(bix2)



class individual:

    def __init__(self, maxStacks):

        self.maxStacks = maxStacks

        # Initialize to random stacks
        n = np.random.randint(2, self.maxStacks + 1)
        self.stacks = np.array([randomStack(1/3, 10) for i in range(n)])

        # Initialize fitness
        self.fitness = -1
        self.fitnessDetails = np.empty(5)

        # Clean up
        self.makeValid(False, [0, 0, 0])
        self.standardize()


    def makeValid(self, filler, bix2):
        # Check that stack sizes (Na) are positive
        # and winding numbers are coprime, fixing if necessary

        for stack in self.stacks:

            # Make sure Na is not below the minimum allowed value of 1
            stack[0] = max(abs(stack[0]), 1)

            # Loop over winding pairs
            for i in range(3):
                n = stack[2*i+1]
                m = stack[2*i+2]

                # Correct all cases where gcd(n,m) != 1
                if n == 0 and m == 0:
                    # (0,0) ==> (1,0), (-1,0), (0,1) or (0,-1)
                    if np.random.rand() < 0.5:
                        stack[2*i+1] = np.random.choice([-1, 1])
                    else:
                        stack[2*i+2] = np.random.choice([-1, 1])

                elif n == 0 and abs(m) > 1:
                    # (0,m), |m|>1 ==> (1,m), (-1,m) or (0,sgn(m))
                    if np.random.rand() < 0.5:
                        stack[2*i+2] /= np.abs(stack[2*i+2])
                    else:
                        stack[2*i+1] = np.random.choice([-1, 1])

                elif abs(n) > 1 and m == 0:
                    # (n,0), |n|>1 ==> (n,1), (n,-1) or (sgn(n),0)
                    if np.random.rand() < 0.5:
                        stack[2*i+1] /= np.abs(stack[2*i+1])
                    else:
                        stack[2*i+2] = np.random.choice([-1, 1])

                elif np.gcd(n, m) > 1:
                    # (n,m) with n,m not coprime. Pick either n,m and
                    # increment or decrement until n,m are coprime
                    direction = np.random.choice([-1, 1])
                    index = 2*i + np.random.choice([1, 2])
                    while np.gcd(stack[2*i+1], stack[2*i+2]) > 1:
                        stack[index] += direction


        if filler:
            # If fillers are added by hand, remove all 'explicit' filler stacks

            XYlist = np.array([getStackXY(stack, bix2) for stack in self.stacks])
            Ylist = XYlist[:, 4:]

            fillers = np.where([YI@YI == 0 for YI in Ylist])[0]

            self.stacks = np.delete(self.stacks, fillers, axis=0)



    def standardize(self):
        # For each stack, use the ambiguity in representing
        # brane stacks with winding numbers to pick a 'standard form'.
        # Flipping the signs of two (n,m) pairs simultaneously does
        # not change the XI, YI. Use this to make as many ns positive as possible.

        for stack in self.stacks:

            # Get ns
            n1, n2, n3 = stack[1::2]

            # Flip pairs 1 and 3
            if n1 < 0 or (n1 == 0 and n2*n3 < 0):
                stack[1:3] *= -1
                stack[5:7] *= -1

            # Flip pairs 2 and 3
            if n2 < 0 or (n2 == 0 and n1*n3 < 0):
                stack[3:7] *= -1

            # Special case (n1,n2,n3)==(0,0,n3<0):
            # Either flip pairs 1 and 3 or pairs 2 and 3
            if n1 == 0 and n2 == 0 and n3 < 0:
                if np.random.rand() < 0.5:
                    stack[1:3] *= -1
                    stack[5:7] *= -1
                else:
                    stack[3:7] *= -1


    def crossOver(self, parent2, xoverProbs):

        # Randomly choose a crossover method
        rnd = np.random.rand()
        cumulative = np.cumsum(xoverProbs)
        method = sum(cumulative < rnd)

        if method == 0:
            # A 1-point crossover method. Pick a random number
            # of rows in each parent and splice together by rows.
            rows1 = np.inf
            rows2 = np.inf

            while rows1 + rows2 > self.maxStacks:
                rows1 = np.random.randint(len(self.stacks)) + 1
                rows2 = np.random.randint(len(parent2.stacks)) + 1

            newStacks = np.zeros([rows1 + rows2, 7], dtype='int')
            newStacks[:rows1] = self.stacks[:rows1]
            newStacks[-rows2:] = parent2.stacks[-rows2:]

        elif method == 1:
            # A 1-point crossover method where the child
            # has the same number of stacks as parent1 (self).
            newStacks = self.stacks.copy()
            rows = np.random.randint(1, min(len(parent2.stacks)+1, len(self.stacks)))

            if np.random.rand() < 0.5:
                newStacks[:rows] = parent2.stacks[:rows]
            else:
                newStacks[-rows:] = parent2.stacks[-rows:]

        elif method == 2:
            # Same as method 0 except that the splice occurs in the middle of a stack
            rows1 = np.inf
            rows2 = np.inf
            col = np.random.randint(1, 7)

            while rows1 + rows2 + 1 > self.maxStacks or rows1 + rows2 + 1 < 2:
                rows1 = np.random.randint(len(self.stacks))
                rows2 = np.random.randint(len(parent2.stacks))

            newStacks = np.zeros([rows1 + rows2 + 1, 7], dtype='int')
            newStacks[:rows1] = self.stacks[:rows1]
            if rows2 > 0:
                newStacks[-rows2:] = parent2.stacks[-rows2:]
            newStacks[rows1, :col] = self.stacks[rows1, :col]
            newStacks[rows1, col:] = parent2.stacks[-(rows2+1), col:]


        elif method == 3:
            # Same as method 1 except that the splice occurs in the middle of a stack
            newStacks = self.stacks.copy()
            rows = np.random.randint(min(len(parent2.stacks), len(self.stacks)))
            col = np.random.randint(1, 7)

            if np.random.rand() < 0.5:
                newStacks[:rows] = parent2.stacks[:rows]
                newStacks[rows, :col] = parent2.stacks[rows, :col]
            else:
                newStacks[-(rows+1):] = parent2.stacks[-(rows+1):]
                newStacks[-(rows+1), :col] = self.stacks[len(self.stacks)-(rows+1), :col]

        else:
            # Clone parent1 (self)
            newStacks = self.stacks.copy()


        # Create child with newStacks
        child = individual(self.maxStacks)
        child.stacks = newStacks
        return child



    def computeFitness(self, bix2, UI, filler, weights, tadScale, susyScale):

        # Only update if not previously computed
        if self.fitness == -1:
            
            Ns = self.stacks[:, 0]

            XYlist = np.array([getStackXY(stack, bix2) for stack in self.stacks])
            Xlist = XYlist[:, :4]
            Ylist = XYlist[:, 4:]

            # Filler branes have all YI=0
            nFiller = np.sum([(YI == 0).all() for YI in Ylist])


            # Tadpole condition: these four integers should be zero
            tadpoles = np.dot(Ns, Xlist) - 8

            # When filler branes are added by hand, the tadpole conditions
            # become inequalities (*not exactly true for tilted tori)
            if filler:
                # For tilted tori filler branes contribute (1+2b1)(1+2b2)(1+2b3) to X0 tadpole
                if tadpoles[0] < 0:
                    tadpoles[0] = -(-tadpoles[0] % np.product(1 + bix2))

                # For tilted tori filler branes contribute 2 to Xi tadpoles
                for i in range(3):
                    if tadpoles[i+1] < 0:
                        tadpoles[i+1] = -(-tadpoles[i+1] % (1 + bix2[i]))


            # K-theory constraint: these four integers should be even
            Kth = np.dot(Ns, Ylist) % 2


            # U0dual = U1*U2*U3, U1dual = U0*U2*U3, etc.
            UIdual = np.array(np.product(UI) / UI, dtype='int')

            # SUSY conditions
            # sum(U^I X^I) must be positive for each stack
            susyX = np.array([np.minimum(0, XI @ UI) for XI in Xlist]) / np.sum(UI)

            # sum(Y^I/U^I)~sum(Y^I U^J U^K U^L) must be zero for each stack
            # Using UIdual avoids fp errors if identically zero
            susyY = np.array([YI @ UIdual for YI in Ylist]) / np.sum(UIdual)


            # Rewards: each lies in the interval [0,1]
            TReward = (1 + np.mean(abs(tadpoles))/tadScale) ** (-1)
            KReward = np.sqrt(1-np.mean(Kth))
            SReward = np.min((1 + (abs(susyX) + abs(susyY))/susyScale) ** (-1))
            MSSMReward = MSSM(Ns, Xlist, Ylist, bix2)

            self.rewards = np.array([TReward, KReward, SReward, MSSMReward])

            # Fitness is weighted sum of rewards
            self.fitness = weights @ self.rewards


            # Record which consistency conditions are satisfied
            self.consCond = ''

            if max(abs(tadpoles)) == 0:
                self.consCond += 'T'
            if max(Kth) == 0:
                self.consCond += 'K'
            if max(abs(susyX)) + max(abs(susyY)) == 0:
                self.consCond += 'S'


            self.tadpoles = tadpoles
            self.Kth = Kth
            self.Xlist = Xlist
            self.Ylist = Ylist

            # Record details of fitness
            self.fitnessDetails = [TReward, KReward, SReward, MSSMReward, nFiller/len(self.stacks)]



    def mutate(self, mutRates, bix2):

        # Extract mutation parameters.
        # The first row contains the probabilities for mutations
        #   which change the number of stacks.
        mutStckSplt, mutStckAdd, mutStckComb, mutStckDel, mutStckPerm = mutRates[:5]
        # The following are not the probabilities of a mutation
        #   to be applied to each winding number, say, but rather
        #   the expected number of mutations on the whole chromosome
        #   (using that X ~ Binom(k,p) has E[X]=kp)
        #   I expect this captures the scaling with nStacks.
        mutNaPM, mutWindPM, mutWindPerm, mutWindSgns = mutRates[5:]


        # Split stack with Na > 1 into two
        if len(self.stacks) < self.maxStacks and np.random.rand() < mutStckSplt:

            splittable = np.where(self.stacks[:, 0] > 1)[0]
            if len(splittable) > 0:
                a = np.random.choice(splittable)
                b = np.random.randint(len(self.stacks) + 1)
                Na = self.stacks[a, 0]
                self.stacks = np.insert(self.stacks, b, self.stacks[a], axis=0)
                self.stacks[a, 0] = np.random.randint(1, Na)
                self.stacks[b, 0] = Na - self.stacks[a, 0]


        # Add random stack
        if len(self.stacks) < self.maxStacks and np.random.rand() < mutStckAdd:
            newStack = randomStack(1/3, 10)
            a = np.random.randint(len(self.stacks) + 1)
            self.stacks = np.insert(self.stacks, a, newStack, axis=0)


        # Combine two stacks with identical winding numbers
        if len(self.stacks) > 2 and np.random.rand() < mutStckComb:

            pairs = np.empty([0, 2], dtype='int')
            for a in range(len(self.stacks)):
                for b in range(a+1, len(self.stacks)):
                    if (self.stacks[a, 1:] == self.stacks[b, 1:]).all():
                        pairs = np.append(pairs, [[a, b]], axis=0)

            if len(pairs) > 0:
                pair = pairs[np.random.randint(len(pairs))]
                a, b = np.random.permutation(pair)
                self.stacks[a, 0] += self.stacks[b, 0]
                self.stacks = np.delete(self.stacks, b, axis=0)


        # Delete stack
        if len(self.stacks) > 2 and np.random.rand() < mutStckDel:
            a = np.random.randint(len(self.stacks))
            self.stacks = np.delete(self.stacks, a, axis=0)



        # Mutations to each stack
        for stack in self.stacks:

            # Change Na
            if np.random.rand() < mutNaPM / len(self.stacks):
                stack[0] += np.random.choice([-1, 1])

            # Loop through stack's winding numbers
            for i in range(1, 7):
                if np.random.rand() < mutWindPM / (6*len(self.stacks)):
                    stack[i] += np.random.choice([-2, -1, 1, 2])
        
            # Permute X,Y
            if np.random.rand() < mutWindPerm / len(self.stacks):
                newStack = np.zeros(7)
                newStack[0] = stack[0]

                if np.random.rand() < 0.75:

                    ind = np.random.randint(3)

                    for i in range(3):

                        if i == ind:
                            newStack[2*i+1] = -stack[2*i+1]
                            newStack[2*i+2] = -stack[2*i+2]
                        else:
                            newStack[2*i+1] = stack[2*i+2] + bix2[i]*(stack[2*i+1] + stack[2*i+2])
                            newStack[2*i+2] = -stack[2*i+1] - bix2[i]*stack[2*i+2]

                stack = newStack

                sigma = np.random.permutation(3)

                for i in range(3):
                    n = stack[2*sigma[i]+1]
                    m = stack[2*sigma[i]+2]

                    if bix2[i] == bix2[sigma[i]]:
                        newStack[2*i+1] = n
                        newStack[2*i+2] = m
                    elif bix2[i] == 0:
                        newStack[2*i+1] = n
                        newStack[2*i+2] = 2*m + n
                    else:
                        newStack[2*i+1] = 2*n
                        newStack[2*i+2] = m - n

                stack = newStack


            # Randomly assign new sign pattern
            if np.random.rand() < mutWindSgns / len(self.stacks):
                # Loop over winding pairs
                for i in range(3):
                    n = stack[2*i+1]
                    m = stack[2*i+2]

                    rnd = np.random.rand()

                    if rnd < 0.25:
                        stack[2*i+1] = -n
                        stack[2*i+2] = m + bix2[i] * n
                    elif rnd < 0.5:
                        stack[2*i+2] = -m - bix2[i] * n
                    elif rnd < 0.75:
                        stack[2*i+1] = -n
                        stack[2*i+2] = -m
    
        # Permute stacks
        if np.random.rand() < mutStckPerm:
            np.random.shuffle(self.stacks)



    def correctSigns(self, bix2):
        # A bit of a cheat: for type A',B',C' branes there is a natural way
        # to change the sign patterns of the winding numbers so that
        # both of the SUSY conditions can potentially be satisfied.
        # Such changes can perhaps be superceded by the 'mutWindSgns' mutation.


        # TO DO:
        # This will have to be adjusted for tilted tori,
        # where it should be mhat that changes sign

        for stack in self.stacks:

            XYlist = getStackXY(stack, bix2)
            Xlist = np.array(XYlist[:4])

            numXzero = np.sum(Xlist == 0)
            numXpos = np.sum(Xlist > 0)

            if numXzero == 0:
                # Type A
                if numXpos == 1:
                    # Flip one of the ms
                    stack[2] *= -1

            elif numXzero == 2:
                # Type B
                if numXpos == 0:
                    # Flip all ns
                    stack[1::2] *= -1
                elif numXpos == 1:
                    if stack[1]*stack[2] != 0:
                        stack[2] *= -1
                    else:
                        stack[4] *= -1
                    XYlist = getStackXY(stack, bix2)
                    XlistNew = np.array(XYlist[:4])
                    if sum(XlistNew > 0) != 2:
                        stack[1::2] *= -1

            elif numXzero == 3:
                # Type C
                if numXpos == 0:
                    # Flip an (n,m) pair
                    stack[5:] *= -1

            elif numXzero == 4:
                # Type D
                pass


    def display(self, bix2):


        print(' Na |  n1  m1  n2  m2  n3  m3   |   X0  X1  X2  X3   |   Y0  Y1  Y2  Y3\n' + '-'*72)

        for i in range(len(self.stacks)):
            stack = self.stacks[i]
            print('{:3d} |'.format(stack[0]), end='')
            for s in stack[1:]:
                print('{:4d}'.format(s), end='')
            print('   | ', end='')
            for XI in self.Xlist[i]:
                if XI == 0:
                    print(' '*4, end='')
                else:
                    print('{:4d}'.format(XI), end='')
            print('   | ', end='')
            for YI in self.Ylist[i]:
                if YI == 0:
                    print(' '*4, end='')
                else:
                    print('{:4d}'.format(YI), end='')
            print('')
        print(' '*34 + '-'*38 + '\n' + ' '*34, end='')
        for tad in self.tadpoles:
            print('%4d' % tad, end='')
        print('   | ', end='')
        for k in self.Kth:
            print('%4d' % k, end='')
        print('')


def randomStack(p, mu):

    # Random Na and winding numbers
    s = np.zeros(7, dtype='int')
    s[0] = np.random.geometric(p)
    s[1:] = randomSkellam(mu, 6)

    return s


def randomSkellam(mu, n):
    # If Y,Z~Pois(mu), then X=Y-Z follows the distribution X~Skellam(mu,mu).
    # This discrete probability distribution has support
    # on all integers, mean zero and variance 2*mu
    pois1 = np.random.poisson(mu, size=n)
    pois2 = np.random.poisson(mu, size=n)

    return pois1 - pois2


def getStackXY(stack, bix2):

    # Extract winding numbers and bix2
    n1, m1, n2, m2, n3, m3 = stack[1:]
    b1x2, b2x2, b3x2 = bix2

    m1hat = (1 + b1x2) * m1 + b1x2 * n1
    m2hat = (1 + b2x2) * m2 + b2x2 * n2
    m3hat = (1 + b3x2) * m3 + b3x2 * n3

    X0 = + n1    * n2    * n3
    X1 = - n1    * m2hat * m3hat
    X2 = - m1hat * n2    * m3hat
    X3 = - m1hat * m2hat * n3

    Y0 = + m1hat * m2hat * m3hat
    Y1 = - m1hat * n2    * n3
    Y2 = - n1    * m2hat * n3
    Y3 = - n1    * n2    * m3hat

    return X0, X1, X2, X3, Y0, Y1, Y2, Y3



def saveSort(stacks):
    # Sort stacks in lexicagraphical order for saving
    order = np.lexsort(stacks.T[::-1])[::-1]
    return stacks[order]



def MSSM(Ns, Xlist, Ylist, bix2):

    #    GAUGE GROUP
    # -----------------
    # Each stack contributes a gauge group factor:
    #   - Type A,B : U(N)
    #   - Type C   : USp(N)
    # U(1) factors are generically anomolous and gain a mass
    # A massless combination U(1) = sum(x_a U(1)_a) requires sum(x_aN_aY_a^I) = 0


    # Identify type C (filler) branes
    filler = [YI@YI == 0 for YI in Ylist]
    notfiller = [not b for b in filler]

    # Find U(4) factors
    U4inds = np.where((Ns == 4) * (notfiller))[0]
    nU4 = len(U4inds)

    # Find U(3) factors
    U3inds = np.where((Ns == 3) * (notfiller))[0]
    nU3 = len(U3inds)

    # Find U(2) factors
    U2inds = np.where((Ns == 2) * (notfiller))[0]
    nU2 = len(U2inds)

    # Find U(1) factors
    U1inds = np.where((Ns == 1) * (notfiller))[0]
    nU1 = len(U1inds)

    # USp(1) ( ~ SU(2) ) factors
    Sp1inds = np.where((Ns == 1) * filler)[0]
    nSp1 = len(Sp1inds)


    # Distances to several gauge groups of interest
    #   - U(3) x U(2) x U(1) x U(1)
    #   - U(3) x Sp(2) x U(1) x U(1)
    #   - U(4) x U(2) x U(2)
    U3U2U1U1dist  = max(0, 1-nU3) + max(0, 1-nU2) + max(0, 2-nU1)
    # U3Sp2U1U1dist = max(0, 1-nU3) + max(0, 1-nSp1) + max(0, 2-nU1)
    # U4U2U2dist    = max(0, 1-nU4) + max(0, 2-nU2)

    gaugeGpBonus = 1 - U3U2U1U1dist/4.0

    chiralDistBest = np.inf

    if U3U2U1U1dist == 0:
        # Check spectrum

        for iU3 in U3inds:
            for iU2 in U2inds:
                quarkMult = Xlist[iU3] @ Ylist[iU2] - Ylist[iU3] @ Xlist[iU2]

                chiralDist = np.abs(quarkMult - 3)

                if chiralDist < chiralDistBest:
                    chiralDistBest = chiralDist

    # if U3Sp2U1U1dist == 0:
    #     # Check spectrum
    #     1

    # if U4U2U2dist == 0:
    #     # Check spectrum
    #     1

    MSSMbonus = 0.5 * gaugeGpBonus + 0.5 * 1/(1 + chiralDistBest)


    return MSSMbonus


def Iab(stacka, stackb, bix2):

    nai = stacka[1::2]
    mai = stacka[2::2]

    nbi = stackb[1::2]
    mbi = stackb[2::2]

    factors = nai*mbi - mai*nbi

    return factors[0] * factors[1] * factors[2]


def saveSolutions(stacks, filePath):

    try:
        # Load previously saved
        loaded = np.load(filePath)
        toSave = np.append(loaded, stacks, axis=0)
    except IOError:
        # No such filed exists
        toSave = stacks
    
    # Save to file
    np.save(filePath, toSave)
