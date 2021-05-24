import numpy as np
from fastprogress import master_bar, progress_bar
from sympy import binomial



def GA(nRuns, numGens, popSize, numSurvive,
       numStacks, bix2, minNa, weights, xoverProb, mutRates,
       progress, save, filePath=None):
    
    for n in range(nRuns):
        print('Run %2d' % (n+1))

        firstFound = False
        consistentStacks = np.empty([0, numStacks, 7], dtype='int')

        # Initialize random population
        pop = population(popSize, numStacks, bix2, minNa, weights)


        if progress:

            # Keep track of fitness distribution and quantiles
            fitnessDist = np.zeros([numGens + 1, popSize])
            fitnessQuantiles = np.zeros([numGens + 1, 4])
            bestRewards = np.zeros([numGens + 1, 3])

            allFits = pop.getAllFits()
            fitnessDist[0] = allFits
            fitnessQuantiles[0] = np.quantile(pop.getAllFits(), [0.25, 0.50, 0.75, 1.00])
            bestRewards[0] = pop.individuals[np.argsort(allFits)[-1]].fitnessDetails[:3]

            # Progress bar for monitoring fitness distribution over the generations
            progBar = master_bar(range(1, numGens + 1))
            progBar.names = ['', '', 'tad', 'K-th', 'SUSY', 'max', 'Q75', 'Q50', 'Q25']
            
            # Progress bar serves as iterator
            gIter = progBar

        else:
            # Iterator
            gIter = range(1, numGens + 1)


        for g in gIter:

            # Breed next generation (this is quick)
            pop.breed(numSurvive, xoverProb, mutRates)

            if progress:
                # Progress bar serves as iterator
                iIter = progress_bar(range(popSize), parent=progBar)
            else:
                # Iterator
                iIter = range(popSize)

            # Compute fitness for all new individuals of population
            for i in iIter:
                pop.individuals[i].updateFitness()

            if len(pop.getConsistent()) > 0:
                if not firstFound:
                    firstFound = True
                    print('\tFirst found at generation : %4d' % g)

                consistentStacks = np.append(consistentStacks,
                                             [e.stacks for e in pop.getConsistent()], axis=0)

            if progress:
                # Update fitness summary statistics
                allFits = pop.getAllFits()
                fitnessDist[g] = allFits
                fitnessQuantiles[g] = np.quantile(allFits, [0.25, 0.50, 0.75, 1.00])
                bestRewards[g] = pop.individuals[np.argsort(allFits)[-1]].fitnessDetails[:3]

                # Update graph of fitness summary statistics
                progBar.update_graph([
                    [[0, g], [1, 1]],
                    [[1.05*g, 2.05*g], [1, 1]],
                    [np.arange(1.05*g, 2.05*g+1), bestRewards[:g+1, 0]],
                    [np.arange(1.05*g, 2.05*g+1), bestRewards[:g+1, 1]],
                    [np.arange(1.05*g, 2.05*g+1), bestRewards[:g+1, 2]],
                    [np.arange(g+1), fitnessQuantiles[:g+1, 3]],
                    [np.arange(g+1), fitnessQuantiles[:g+1, 2]],
                    [np.arange(g+1), fitnessQuantiles[:g+1, 1]],
                    [np.arange(g+1), fitnessQuantiles[:g+1, 0]]
                ], [-0.1*g, 2.3*g], [0, 1.05], figsize=(14, 8))

        if firstFound:
            print('\t' + ' '*10 + 'Solutions found : %4d' % len(consistentStacks))

            consistentStacks = np.unique(consistentStacks, axis=0)

            print('\t' + ' '*3 + 'Unique solutions found : %4d' % len(consistentStacks))

        if save:
            saveSolutions(consistentStacks, filePath)

        print('')




class population:
    def __init__(self, size, numStacks, bix2, minNa, weights):
        self.size = size

        self.individuals = [individual(numStacks, bix2, minNa, weights) for i in range(size)]
        self.individuals = np.array(self.individuals)
        for e in self.individuals:
            e.updateFitness()


    def breed(self, numSurvive, xoverProb, mutRates):

        # Create empty array for next generation and copy
        # over 'numSurvive' of the fittest individuals
        newIndividuals = np.empty([self.size], dtype='object')
        newIndividuals[:numSurvive] = self.getFittest(numSurvive)

        # Loop until next generation is full
        for i in range(numSurvive, self.size):

            # Select two parents
            parent1 = self.binaryTournament()
            parent2 = self.binaryTournament()

            # Form child either by crossover or by cloning
            if np.random.rand() < xoverProb:
                child = parent1.crossOver(parent2)
            else:
                child = parent1.clone()

            # Alter child
            child.mutate(mutRates)
            child.makeValid()
            child.standardize()

            # Add child to next generation
            newIndividuals[i] = child

        # Update population to newly created generation
        self.individuals = newIndividuals


    def binaryTournament(self):
        # Select two random individuals and return the fittest
        i, j = np.random.randint(self.size, size=2)

        if self.individuals[i].fitness > self.individuals[j].fitness:
            return self.individuals[i]
        else:
            return self.individuals[j]


    def getAllFits(self):
        # Returns list of the fitness for all individuals in the population
        return [e.fitness for e in self.individuals]


    def getFittest(self, n):
        # Returns the n fittest individuals in the population
        fits = self.getAllFits()
        bestn = np.argsort(fits)[-1:-(n+1):-1]
        return self.individuals[bestn]

    def getConsistent(self):
        return self.individuals[[e.isConsistent() for e in self.individuals]]

    def displayFittest(self, n):
        # Display the n fittest individuals in the population
        for e in self.getFittest(n):
            e.display()



class individual:
    def __init__(self, numStacks, bix2, minNa, weights):
        self.numStacks = numStacks
        self.bix2 = bix2
        self.minNa = minNa
        self.weights = weights

        # Initialize to random stacks
        self.stacks = np.array([randomStack() for i in range(numStacks)])

        # Initialize fitness
        self.fitness = -1
        self.fitnessDetails = np.empty([8], dtype='object')

        # Clean up
        self.makeValid()
        self.standardize()


    def updateFitness(self):

        # Only update if not previously computed
        if self.fitness is -1:
            
            Ns = self.stacks[:, 0]

            XYlist = np.array([getStackXY(stack, self.bix2) for stack in self.stacks])
            Xlist = XYlist[:, :4]
            Ylist = XYlist[:, 4:]

            # Tadpole condition: these four integers should be zero
            tadpoles = np.dot(Ns, Xlist) - 8

            # K-theory constraint: these four integers should be even
            Kth = np.dot(Ns, Ylist) % 2


            if max(abs(tadpoles)) < 16:
                # SUSY condition: evaluation results
                uBest, Xterms, Yterms = SUSY(Ns, XYlist)
            else:
                uBest = [0, 0, 0, 0]
                Xterms = np.inf * np.ones(len(Ns))
                Yterms = np.inf * np.ones(len(Ns))


            # Rewards: each lies in the interval [0,1]
            tadReward = np.mean((1 + abs(tadpoles)/8) ** (-1))
            KthReward = np.sqrt(1-sum(Kth)/4)
            susyReward = np.mean((1 + abs(Xterms) + abs(Yterms)) ** (-1))


            # Record details of fitness
            self.fitnessDetails = [tadReward, KthReward, susyReward,
                                   tadpoles, Kth, uBest, Xterms, Yterms]

            # Fitness is weighted sum of rewards
            self.fitness = + self.weights[0] * tadReward \
                           + self.weights[1] * KthReward \
                           + self.weights[2] * susyReward


            # # If the model is consistent, add also an MSSM bonus
            # if self.isConsistent():
            #     MSSMbonus = MSSM(Ns, Xlist, Ylist)
            #     self.fitness += MSSMbonus



    def clone(self):

        # Create child with same Na & winding numbers
        child = individual(self.numStacks, self.bix2, self.minNa, self.weights)
        child.stacks = self.stacks

        return child


    def crossOver(self, parent2):

        #     METHOD 1
        # ----------------
        # A 1-point crossover method. Pick a random crossover point in the array
        # of Na and winding numbers and splice together, scanning either by rows or by cols
        col = np.random.randint(7)
        row = np.random.randint(self.numStacks)

        newStacks = self.stacks.copy()
        if np.random.rand() < 0.5:
            newStacks[row, col:] = parent2.stacks[row, col:]
            newStacks[(row+1):] = parent2.stacks[(row+1):]
        else:
            newStacks[row:, col] = parent2.stacks[row:, col]
            newStacks[:, (col+1):] = parent2.stacks[:, (col+1):]


        # #     METHOD 2
        # # ----------------
        # # A 1-point crossover method. Pick a random row and splice together by rows
        # row = np.random.randint(1, self.numStacks)

        # newStacks = self.stacks.copy()
        # newStacks[row:] = parent2.stacks[row:]


        # #     METHOD 3
        # # ----------------
        # # A uniform crossover method. Whether each row comes
        # # from parent 1 or 2 is determined at random.
        # newStacks = self.stacks.copy()
        # randRows = np.random.choice([False, True], self.numStacks)
        # newStacks[randRows] = parent2.stacks[randRows]


        # Create child and set Na & winding numbers
        child = individual(self.numStacks, self.bix2, self.minNa, self.weights)
        child.stacks = newStacks

        return child


    def mutate(self, mutRates):

        # Extract various (relative) mutation rates
        mutWindBnml, mutWindZero, mutWindSign = mutRates[0:3]
        mutPairRand, mutPairPerm, mutPairSign = mutRates[3:6]
        mutStckRand, mutSizeBnml              = mutRates[6:8]

        # Loop through all stacks
        for stack in self.stacks:

            # Loop through all winding numbers
            for i in range(1, 7):

                # Add (centered) binomial random variable to random winding
                if np.random.rand() < mutWindBnml:
                    stack[i] += randomBnml(3, 1)
                
                # Set winding number to zero
                if np.random.rand() < mutWindZero:
                    stack[i] = 0

                # Flip sign of winding number
                if np.random.rand() < mutWindSign:
                    stack[i] *= -1
        
            # Loop through all winding pairs
            for i in range(3):
                # Set winding pair to random binomials
                if np.random.rand() < mutPairRand:
                    stack[2*i+1:2*i+3] += randomBnml(3, 2)

                # Flip sign of winding pair
                if np.random.rand() < mutPairSign:
                    stack[2*i+1:2*i+3] *= -1
                            
            # Randomly permute winding pairs
            if np.random.rand() < mutPairPerm:
                i, j, k = np.random.choice(range(3), size=3, replace=False)
                temp = stack[2*i+1:2*i+3]
                stack[2*i+1:2*i+3] = stack[2*j+1:2*j+3]
                stack[2*j+1:2*j+3] = stack[2*k+1:2*k+3]
                stack[2*k+1:2*k+3] = temp
    
            # Replace with random stack
            if np.random.rand() < mutStckRand:
                stack = randomStack()

            # Change N_a
            if np.random.rand() < mutSizeBnml:
                stack[0] += randomBnml(5, 1)


    def makeValid(self):

        # Loop over stacks
        for stack in self.stacks:

            # Make sure Na is not below the minimum allowed value (0 or 1)
            stack[0] = max(stack[0], self.minNa)

            # Loop over winding pairs
            for i in range(3):
                n = stack[2*i+1]
                m = stack[2*i+2]

                # Correct cases where gcd(n,m) != 1
                if n == 0 and m == 0:
                    # (0,0) ==> (1,0), (-1,0), (0,1) or (0,-1)
                    if np.random.rand() < 0.5:
                        stack[2*i+1] = np.random.choice([-1, 1])
                    else:
                        stack[2*i+2] = np.random.choice([-1, 1])

                elif n == 0 and abs(m) > 1:
                    # (0,m), |m|>1 ==> (1,m), (-1,m) or (0,sgn(m))
                    if np.random.rand() < 0.5:
                        stack[2*i+1] = np.random.choice([-1, 1])
                    else:
                        stack[2*i+2] /= np.abs(stack[2*i+2])

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
                    index = 2 * i + np.random.choice([1, 2])
                    while np.gcd(stack[2*i+1], stack[2*i+2]) > 1:
                        stack[index] += direction


    def standardize(self):

        # For each stack, flip signs of the first two winding pairs
        # so that (n1>0,m1) or (0,m1>0) and also (n2>0,m2) or (0,m2>0).
        # Doing so does not change any Xs or Ys
        for stack in self.stacks:
            n1, m1, n2, m2 = stack[1:5]
            if n1 < 0 or (n1 == 0 and m1 < 0):
                stack[1:3] *= -1
                stack[5:7] *= -1
            if n2 < 0 or (n2 == 0 and m2 < 0):
                stack[3:7] *= -1


    def isConsistent(self):

        # Check if all consistency conditions are satisfied
        tadCancelled = (self.fitnessDetails[3] == [0, 0, 0, 0]).all()
        KthCondition = (self.fitnessDetails[4] == [0, 0, 0, 0]).all()
        susyCondition = (np.max(np.abs(self.fitnessDetails[6:])) == 0)

        return tadCancelled and KthCondition and susyCondition

        # return (self.fitness >= 1)


    def display(self):
        hdr1 = ["N_a", "n1", "m1", "n2", "m2", "n3", "m3"]
        hdr2 = ["X0", "X1", "X2", "X3", "Y0", "Y1", "Y2", "Y3"]
        hdr3 = ["SUSY-X", "SUSY-Y"]
        frmt1 = ("{:>5} |" + "{:>4}"*6)
        frmt2 = ("{:>4}"*4 + " "*5 + "{:>4}"*4)
        frmt3 = ("{:>6}" + " "*5 + "{:>6}")

        tadReward, KthReward, susyReward = self.fitnessDetails[:3]
        tadpoles, Kth                    = self.fitnessDetails[3:5]
        uBest, Xterms, Yterms            = self.fitnessDetails[5:]

        argsTad = ["tadpole", self.weights[0]*tadReward, self.weights[0], tadReward]
        argsKth = ["K-theory", self.weights[1]*KthReward, self.weights[1], KthReward]
        argsSUSY = ["SUSY", self.weights[2]*susyReward, self.weights[2], susyReward]
        frmtTad = ("\n{:>10} = {:.4f}\t= {:.2f} x {:.4f}")
        frmtKth = ("{:>10} = {:.4f}\t= {:.2f} x {:.2f}")
        frmtSUSY = ("{:>10} = {:.4f}\t= {:.2f} x {:.4f}")
        frmtXY = ("{:7.4f}" + " "*5 + "{:7.4f}")

        # Headers
        print("\n", frmt1.format(*hdr1), end=' '*5)
        print(frmt2.format(*hdr2), end=' '*5)
        print(frmt3.format(*hdr3))
        print('-'*98)

        # Na, (n,m), XI, YI for each stack
        for stack, i in zip(self.stacks, range(self.numStacks)):
            print(frmt1.format(*stack), end=' '*5)
            print(frmt2.format(*getStackXY(stack, self.bix2)), end=' '*5)
            print(frmtXY.format(Xterms[i], Yterms[i]))

        # Rewards and fitness details
        print(frmtTad.format(*argsTad), end='\t\t')
        print(("{:>4}"*4).format(*tadpoles))
        print(frmtKth.format(*argsKth), end='\t\t')
        print(("{:>4}"*4).format(*Kth))
        print(frmtSUSY.format(*argsSUSY), end='\t\t')
        print(("{:8.4f}"*4).format(*uBest), end='\n')
        print(("{:>10} = {:.4f}").format(*["fitness", self.fitness]), end='\t\t\t\t')
        print(("{:8.2f}"*4).format(*uBest/min(np.abs(uBest))), end='\n\n')


def randomStack():

    # Random Na and winding numbers
    s = np.zeros(7, dtype='int')
    s[0] = np.random.geometric(0.5)
    s[1:] = randomBnml(20, 6)

    return s


def randomBnml(cap, n):
    # Maybe change this so that it has support on all integers?
    return np.random.binomial(2*cap, 0.5, size=n) - cap


def getStackXY(stack, bix2):

    # Extract winding numbers and bi
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


def SUSY(Ns, XYlist):
    # The SUSY (in)equalities are overconstrained. Find solution
    # for a subset of stacks and see how well it extends to the rest.

    # Threshold for ratios: proposed solutions are thrown out
    # if U^I/U^J < eps for any I,J=0,1,2,3
    eps = 10**(-4)

    # Extract Xs and Ys
    Xlist = XYlist[:, :4]
    Ylist = XYlist[:, 4:]


    # - Filler branes have all YI=0 and the SUSY Y-term is automatically solved.
    # - Stacks with exactly one nonzero YI cannot possibly satisfy the SUSY Y-term.
    # - Stacks with two or more nonzero YI can be used to solve for the UI quickly.
    #     These will lead to a non-positive UI unless they have
    #     one or more positive YI and one or more negative YI.

    # For each stack find how many YI positive and how many are negative
    posYcounts = np.sum(Ylist > 0, axis=1)
    negYcounts = np.sum(Ylist < 0, axis=1)

    # Find all of these 'constraining' stacks
    constStacks = np.where((posYcounts > 0) * (negYcounts > 0) * (Ns > 0))[0]

    # Get list of unique constraints (some may be redundant)
    Yconst = Ylist[constStacks]
    Yconst = np.array([YI / (np.gcd.reduce(YI) * np.sign(np.sum(YI) + 0.5))
                       for YI in Yconst], dtype='int')
    if len(Yconst) > 0:
        Yconst = np.unique(Yconst, axis=0)


    # Initialize vars for best SUSY solution
    valBest = np.inf
    uBest = np.zeros(4)
    XtermsBest = np.inf * np.ones(len(Ns))
    YtermsBest = np.inf * np.ones(len(Ns))


    # If there are three or more constraining stacks,
    # pick three at random and solve for the UI exactly.
    if len(Yconst) >= 3:
        # print('Three or more constraints.')

        # Repeat a few times and pick the best
        for r in range(min(4, binomial(len(Yconst), 3))):

            # Pick three random constraining stacks and form matrix of Yi
            inds = np.random.choice(range(len(Yconst)), size=3, replace=False)
            matrix = Yconst[inds, 1:]

            if np.linalg.det(matrix) != 0:
                # If this matrix can be inverted, solve for the Ui (having set U0=1)
                uiinv = np.dot(np.linalg.inv(matrix), -Yconst[inds, 0])
                uIinv = np.block([1, uiinv])

                if min(uIinv) / max(uIinv) < eps:
                    # Moduli must not be negative and the ratios
                    # must not be too large or too small
                    continue

                # We have a valid solution for UI>0 and the ratios
                # not too extreme. Now normalize to lie on S^3
                uI = 1/uIinv
                uI /= np.sqrt(uI@uI)

                # sum(U^I X^I, I) must be positive for each stack
                Xterms = np.array([np.minimum(0, np.sign(Ns[a]) * Xlist[a] @ uI)
                                   for a in range(len(Xlist))])
                # sum(Y^I/U^I, I) must be zero for each stack
                # (three are automatically zero, by construction)
                Yterms = np.array([np.sign(Ns[a]) * Ylist[a] @ (1/uI) for a in range(len(Ylist))])

                # Measure of how terribly this proposed solution for UI extends to all stacks
                val = Xterms@Xterms + Yterms@Yterms

                # Keep track of the best solution
                if val < valBest:
                    valBest = val
                    uBest = uI
                    XtermsBest = Xterms
                    YtermsBest = Yterms


    # If there are two constraining stacks, or if the exact solutions
    # with three or more are all invalid, then there is a 1-parameter
    # family of UI to scan over
    # if (len(Yconst) >= 2) and (valBest == np.inf):
    if len(Yconst) == 2:
        # print('Two constraints.')

        # Repeat a few times and pick the best
        for r in range(min(1, binomial(len(Yconst), 2))):

            a, b = np.random.choice(range(len(Yconst)), 2, replace=False)

            i = 1
            matrix = Yconst[[a, b], 2:]
            if np.linalg.det(matrix) == 0:
                i = 2
                matrix = Yconst[[a, b], 1::2]
            if np.linalg.det(matrix) == 0:
                i = 3
                matrix = Yconst[[a, b], 1:3]
            if np.linalg.det(matrix) == 0:
                continue

            matrixInv = np.linalg.inv(matrix)

            for logui in np.linspace(np.log(eps), np.log(1/eps), 20):
                ui = np.exp(logui)
                ujkInv = np.dot(matrixInv, -Yconst[[a, b], 0] - Yconst[[a, b], i] * 1/ui)

                if i is 1:
                    uIinv = np.block([1, 1/ui, ujkInv])
                elif i is 2:
                    uIinv = np.block([1, ujkInv[0], 1/ui, ujkInv[1]])
                elif i is 3:
                    uIinv = np.block([1, ujkInv, 1/ui])


                if min(uIinv) / max(uIinv) < eps:
                    # Moduli must not be negative and the ratios
                    # must not be too large or too small
                    continue

                # We have a valid solution for UI>0 and the ratios
                # not too extreme. Now normalize to lie on S^3
                uI = 1/uIinv
                uI /= np.sqrt(uI@uI)

                # sum(U^I X^I, I) must be positive for each stack
                Xterms = np.array([np.minimum(0, np.sign(Ns[a]) * Xlist[a] @ uI)
                                   for a in range(len(Xlist))])
                # sum(Y^I/U^I, I) must be zero for each stack
                # (three are automatically zero, by construction)
                Yterms = np.array([np.sign(Ns[a]) * Ylist[a] @ (1/uI) for a in range(len(Ylist))])

                # Measure of how terribly this proposed solution for UI extends to all stacks
                val = Xterms@Xterms + Yterms@Yterms

                # Keep track of the best solution
                if val < valBest:
                    valBest = val
                    uBest = uI
                    XtermsBest = Xterms
                    YtermsBest = Yterms

            

    # If these is only one constraining stack, or if the solutions
    # above are all invalid, then there is a 2-parameter family of UI to scan over
    # if (len(Yconst) >= 1) and (valBest == np.inf):
    if len(Yconst) == 1:
        # print('One constraint.')
        # Repeat a few times and pick the best
        for r in range(min(3, len(Yconst))):

            a = np.random.choice(range(len(Yconst)))

            i, j, k = [1, 2, 3]
            if Yconst[a, k] == 0:
                i, j, k = [1, 3, 2]
            if Yconst[a, k] == 0:
                i, j, k = [2, 3, 1]


            for logui in np.linspace(np.log(eps), np.log(1/eps), 10):
                ui = np.exp(logui)

                for loguj in np.linspace(np.log(eps), np.log(1/eps), 10):
                    uj = np.exp(loguj)

                    ukInv = -(Yconst[a, 0] + Yconst[a, i]/ui + Yconst[a, j]/uj) / Yconst[a, k]

                    if k == 3:
                        uIinv = np.array([1, 1/ui, 1/uj, ukInv])
                    elif k == 2:
                        uIinv = np.array([1, 1/ui, ukInv, 1/uj])
                    elif k == 1:
                        uIinv = np.array([1, ukInv, 1/ui, 1/uj])


                    if min(uIinv) / max(uIinv) < eps:
                        # Moduli must not be negative and the ratios
                        # must not be too large or too small
                        continue

                    # We have a valid solution for UI>0 and the ratios
                    # not too extreme. Now normalize to lie on S^3
                    uI = 1/uIinv
                    uI /= np.sqrt(uI@uI)

                    # sum(U^I X^I, I) must be positive for each stack
                    Xterms = np.array([np.minimum(0, np.sign(Ns[a]) * Xlist[a] @ uI)
                                       for a in range(len(Xlist))])
                    # sum(Y^I/U^I, I) must be zero for each stack
                    # (three are automatically zero, by construction)
                    Yterms = np.array([np.sign(Ns[a]) * Ylist[a] @ (1/uI) for a in range(len(Ylist))])

                    # Measure of how terribly this proposed solution for UI extends to all stacks
                    val = Xterms@Xterms + Yterms@Yterms

                    # Keep track of the best solution
                    if val < valBest:
                        valBest = val
                        uBest = uI
                        XtermsBest = Xterms
                        YtermsBest = Yterms


    # No constraining stacks or failure of all of the above
    # No reward?


    return uBest, XtermsBest, YtermsBest



def MSSM(Ns, Xlist, Ylist):

    # Filler branes have all YI=0
    filler = [YI@YI == 0 for YI in Ylist]

    # SU(3) factors
    SU3inds = np.where((Ns == 3) * (not filler))[0]

    # SU(2) factors
    SU2inds = np.where((Ns == 2) * (not filler))[0]

    # USp(1) ( = SU(2) ) factors
    USp1inds = np.where((Ns == 1) * filler)[0]

    gaugeGroupDist = max(0, 1 - len(SU3inds)) + max(0, 1 - len(SU2inds) - len(USp1inds))

    if gaugeGroupDist == 0:
        # Contains SU(3) x SU(2) x U(1)

        # Loop over ways to pick which factors correspond to SM gauge group
        for i3 in SU3inds:
            for i2 in np.block([SU2inds, USp1inds]):
                1


    MSSMbonus = np.exp(-gaugeGroupDist)


    return MSSMbonus


def saveSolutions(stacks, filePath):

    # try:
    #     # load previously saved solutions
    #     saved = np.load(filePath)
    #     # add new solutions and remove duplicates
    #     toSave = np.unique(np.append(saved, stacks, axis=0), axis=0)
    #     newlySaved = len(toSave) - len(saved)
    # except FileNotFoundError:
    #     # no previously saved solutions
    #     # just remove duplicates
    #     toSave = np.unique(stacks, axis=0)
    #     newlySaved = len(toSave)
    
    # # save to file
    # np.save(filePath, toSave)
    
    # print('\t' + ' '*6 + 'New solutions saved : %4d' % newlySaved)

    try:
        saved = np.load(filePath)
        toSave = np.append(saved, len(stacks))
    except FileNotFoundError:
        toSave = np.array([len(stacks)])

    np.save(filePath, toSave)
