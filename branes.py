import numpy as np
from fastprogress import master_bar, progress_bar


class population:
    def __init__(self, size, numStacks, bix2, minNa, weights, filler):
        self.size = size

        self.elements = [element(numStacks, bix2, minNa, weights, filler) for i in range(size)]
        self.elements = np.array(self.elements)
        for e in self.elements:
            e.updateFitness()


    def breed(self, numSurvive, xoverProb, mutRates):

        # create empty array for next generation and copy
        # over 'numSurvive' of the fittest elements
        newElements = np.empty([self.size], dtype='object')
        newElements[:numSurvive] = self.getFittest(numSurvive)

        # loop until next generation is full
        for i in range(numSurvive, self.size):

            # select two parents
            parent1 = self.binaryTournament()
            parent2 = self.binaryTournament()

            # form child by crossover or by cloning
            if np.random.rand() < xoverProb:
                child = parent1.crossOver(parent2)
            else:
                child = parent1.clone()

            # alter child
            child.mutate(mutRates)
            child.makeValid()
            child.standardize()

            # add child to next generation
            newElements[i] = child

        # update population to newly created generation
        self.elements = newElements


    def binaryTournament(self):
        # select two random elements and return the fittest
        i, j = np.random.randint(self.size, size=2)

        if self.elements[i].fitness > self.elements[j].fitness:
            return self.elements[i]
        else:
            return self.elements[j]


    def getAllFits(self):
        # returns list of the fitness for all elements in the population
        return [e.fitness for e in self.elements]


    def getFittest(self, n):
        # returns the n fittest elements in the population
        fits = self.getAllFits()
        bestn = np.argsort(fits)[-1:-(n+1):-1]
        return self.elements[bestn]

    def getConsistent(self):
        return self.elements[[e.isConsistent() for e in self.elements]]

    def displayFittest(self, n):
        # display the n fittest elements in the population
        for e in self.getFittest(n):
            e.display()


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




def GA(nRuns, numGens, popSize, numSurvive, filler,
       numStacks, bix2, minNa, weights, xoverProb, mutRates,
       progress, save, filePath=None):
    
    for n in range(1, nRuns+1):
        print('Run %2d' % n)

        firstFound = False
        consistentStacks = np.empty([0, numStacks, 7], dtype='int')

        # initialize random population
        pop = population(popSize, numStacks, bix2, minNa, weights, filler)


        if progress:

            # keep track of fitness distribution and quantiles
            fitnessDist = np.zeros([numGens + 1, popSize])
            fitnessQuantiles = np.zeros([numGens + 1, 4])
            bestRewards = np.zeros([numGens + 1, 3])

            allFits = pop.getAllFits()
            fitnessDist[0] = allFits
            fitnessQuantiles[0] = np.quantile(pop.getAllFits(), [0.25, 0.50, 0.75, 1.00])
            bestRewards[0] = pop.elements[np.argsort(allFits)[-1]].fitnessDetails[:3]

            # progress bar for monitoring fitness distribution over the generations
            progBar = master_bar(range(1, numGens + 1))
            progBar.names = ['', '', 'tad', 'K-th', 'SUSY', 'max', 'Q75', 'Q50', 'Q25']
            gIter = progBar

        else:
            gIter = range(1, numGens + 1)


        for g in gIter:

            # breed next generation (this is quick)
            pop.breed(numSurvive, xoverProb, mutRates)

            # compute fitness for all new elements of population
            if progress:
                iIter = progress_bar(range(popSize), parent=progBar)
            else:
                iIter = range(popSize)

            for i in iIter:
                pop.elements[i].updateFitness()

            if len(pop.getConsistent()) > 0:
                if not firstFound:
                    firstFound = True
                    print('\tFirst found at generation : %4d' % g)

                consistentStacks = np.append(consistentStacks,
                                             [e.stacks for e in pop.getConsistent()], axis=0)

            if progress:
                # update fitness summary statistics
                allFits = pop.getAllFits()
                fitnessDist[g] = allFits
                fitnessQuantiles[g] = np.quantile(allFits, [0.25, 0.50, 0.75, 1.00])
                bestRewards[g] = pop.elements[np.argsort(allFits)[-1]].fitnessDetails[:3]

                # update graph of fitness summary statistics
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
            consistentStacks = np.unique(consistentStacks, axis=0)

            print('\t' + ' '*10 + 'Solutions found : %4d' % len(consistentStacks))

        if save:
            saveSolutions(consistentStacks, filePath)

        print('')






class element:
    def __init__(self, numStacks, bix2, minNa, weights, filler):
        self.numStacks = numStacks
        self.bix2 = bix2
        self.minNa = minNa
        self.weights = weights
        self.filler = filler

        self.stacks = np.array([randomStack() for i in range(numStacks)])

        self.fitness = -1
        self.fitnessDetails = np.empty([8], dtype='object')

        self.makeValid()
        self.standardize()


    def updateFitness(self):
        if self.fitness is -1:
            
            Ns = self.stacks[:, 0]

            XYlist = np.array([getStackXY(stack, self.bix2) for stack in self.stacks])
            Xlist = XYlist[:, :4]
            Ylist = XYlist[:, 4:]

            # tadpole condition
            tadpoles = np.dot(Ns, Xlist) - 8

            fillerNa = np.zeros(4)

            if self.filler:
                # !!
                # THIS NEEDS TO BE CHANGED FOR NONZERO bix2
                # !!

                # add filler branes to automatically satisfy tadpole cancellation
                fillerNa = [max(0, -tad) for tad in tadpoles]
                Ns = np.append(Ns, fillerNa)


                XYlist = np.append(XYlist, [[1, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0, 0, 0]], axis=0)

                Xlist = XYlist[:, :4]
                Ylist = XYlist[:, 4:]

                # these should now all be zero
                tadpoles = np.dot(Ns, Xlist) - 8


            # K-theory constraint
            Kth = np.dot(Ns, Ylist) % 2


            # SUSY condition
            susyBest, uBest, Xterms, Yterms = SUSY(Ns, XYlist)


            # tadReward = np.mean([(1 + abs(tad/8.0))**(-1) for tad in tadpoles])

            # tadReward = (1 + np.sqrt(tadpoles@tadpoles)/32) ** (-1)

            # tadpolesPosx2 = (3*tadpoles + abs(tadpoles)) / 2
            # tadReward = (1 + np.sqrt(tadpolesPosx2@tadpolesPosx2)/8) ** (-1)
            # tadReward = np.mean((1 + abs(tadpolesPosx2)/8) ** (-1))

            tadpoleDists = abs(tadpoles)
            tadReward = np.mean((1 + tadpoleDists/8) ** (-1))
            # tadReward = np.mean(np.exp(-tadpoleDists/8))

            KthReward = np.sqrt(1-sum(Kth)/4)

            # susyXReward = np.mean((1 + abs(Xterms))**(-1))
            # susyYReward = np.mean((1 + abs(Yterms))**(-1))
            # susyXReward = np.mean(np.exp(-abs(Xterms)))
            # susyYReward = np.mean(np.exp(-abs(Yterms)))
            # susyReward = (susyXReward + susyYReward) / 2

            susyDists = abs(Xterms) + abs(Yterms)
            susyReward = np.mean((1 + susyDists) ** (-1))
            # susyReward = np.mean(np.exp(-susyDists))


            self.fitnessDetails = [tadReward, KthReward, susyReward,
                                   tadpoles, fillerNa, Kth, uBest, Xterms, Yterms]


            self.fitness = + self.weights[0]*tadReward \
                           + self.weights[1]*KthReward \
                           + self.weights[2]*susyReward


            # if self.isConsistent():
            #     MSSMbonus = MSSM(Ns, Xlist, Ylist)
            #     self.fitness += MSSMbonus

            # # sort stacks based on their SUSY distances
            # order = np.argsort(self.susyDists)

            # self.stacks = self.stacks[order]
            # Xterms = Xterms[order]
            # Yterms = Yterms[order]



    def clone(self):

        # create child with same Na & winding numbers
        child = element(self.numStacks, self.bix2, self.minNa, self.weights, self.filler)
        child.stacks = self.stacks

        return child


    def crossOver(self, parent2):

        # choose xover point and splice, either by rows or by cols
        col = np.random.randint(7)
        row = np.random.randint(self.numStacks)

        newStacks = self.stacks.copy()
        if np.random.rand() < 0.5:
            newStacks[row, col:] = parent2.stacks[row, col:]
            newStacks[(row+1):] = parent2.stacks[(row+1):]
        else:
            newStacks[row:, col] = parent2.stacks[row:, col]
            newStacks[:, (col+1):] = parent2.stacks[:, (col+1):]


        # # choose row for xover
        # row = np.random.randint(1, self.numStacks)

        # newStacks = self.stacks.copy()
        # newStacks[row:] = parent2.stacks[row:]



        # create child and set Na & winding numbers
        child = element(self.numStacks, self.bix2, self.minNa, self.weights, self.filler)
        child.stacks = newStacks

        return child


    def mutate(self, mutRates):

        # extract various (relative) mutation rates
        mutWindBnml, mutWindZero, mutWindSign = mutRates[0:3]
        mutPairRand, mutPairPerm, mutPairSign = mutRates[3:6]
        mutStckRand, mutSizeBnml              = mutRates[6:8]

        # loop through all stacks
        for stack in self.stacks:

            # loop through all winding numbers
            for i in range(1, 7):

                # add (centered) binomial random variable to random winding
                if np.random.rand() < mutWindBnml:
                    stack[i] += randomBnml(3, 1)
                
                # set winding number to zero
                if np.random.rand() < mutWindZero:
                    stack[i] = 0

                # flip sign of winding number
                if np.random.rand() < mutWindSign:
                    stack[i] *= -1
        
            # loop through all winding pairs
            for i in range(3):
                # set winding pair to random binomials
                if np.random.rand() < mutPairRand:
                    stack[2*i+1:2*i+3] += randomBnml(3, 2)

                # flip sign of winding pair
                if np.random.rand() < mutPairSign:
                    stack[2*i+1:2*i+3] *= -1
                            
            # randomly permute winding pairs
            if np.random.rand() < mutPairPerm:
                i, j, k = np.random.choice(range(3), size=3, replace=False)
                temp = stack[2*i+1:2*i+3]
                stack[2*i+1:2*i+3] = stack[2*j+1:2*j+3]
                stack[2*j+1:2*j+3] = stack[2*k+1:2*k+3]
                stack[2*k+1:2*k+3] = temp
    
            # replace with random stack
            if np.random.rand() < mutStckRand:
                stack = randomStack()

            # change N_a
            if np.random.rand() < mutSizeBnml:
                stack[0] += randomBnml(5, 1)


    def makeValid(self):

        for stack in self.stacks:
            stack[0] = max(stack[0], self.minNa)

            # loop over winding pairs
            for i in range(3):
                n = stack[2*i+1]
                m = stack[2*i+2]

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

        # For each stack, flip signs of two winding pairs
        # so that (n1>0,m1) or (0,m1>0) and also (n2>0,m2) or (0,m2>0)
        # Doing so does not change any Xs or Ys
        for stack in self.stacks:
            n1, m1, n2, m2 = stack[1:5]
            if n1 < 0 or (n1 == 0 and m1 < 0):
                stack[1:3] *= -1
                stack[5:7] *= -1
            if n2 < 0 or (n2 == 0 and m2 < 0):
                stack[3:7] *= -1

        # # Sort stacks in lexicographical order based on Xs
        # Xlist = np.array([getStackXY(stack, self.bix2)[:4]
        #                   for stack in self.stacks])
        # X0s = Xlist[:, 0]
        # X1s = Xlist[:, 1]
        # X2s = Xlist[:, 2]
        # X3s = Xlist[:, 3]

        # order = np.arange(len(Xlist))
        # done = False

        # while not done:
        #     done = True
        #     for a in range(1, len(Xlist)):
        #         swap = False
        #         if X0s[order[a-1]] < X0s[order[a]]:
        #             swap = True
        #         elif X0s[order[a-1]] == X0s[order[a]]:
        #             if X1s[order[a-1]] < X1s[order[a]]:
        #                 swap = True
        #             elif X1s[order[a-1]] == X1s[order[a]]:
        #                 if X2s[order[a-1]] < X2s[order[a]]:
        #                     swap = True
        #                 elif X2s[order[a-1]] == X2s[order[a]]:
        #                     if X3s[order[a-1]] < X3s[order[a]]:
        #                         swap = True
        #         if swap:
        #             done = False
        #             temp = order[a]
        #             order[a] = order[a-1]
        #             order[a-1] = temp


    def isConsistent(self):
        tadCancelled = (self.fitnessDetails[3] == [0, 0, 0, 0]).all()
        KthCondition = (self.fitnessDetails[5] == [0, 0, 0, 0]).all()
        susyCondition = (np.max(np.abs(self.fitnessDetails[7:])) == 0)

        return tadCancelled and KthCondition and susyCondition


    def display(self):
        hdr1 = ["N_a", "n1", "m1", "n2", "m2", "n3", "m3"]
        hdr2 = ["X0", "X1", "X2", "X3", "Y0", "Y1", "Y2", "Y3"]
        hdr3 = ["SUSY-X", "SUSY-Y"]
        frmt1 = ("{:>5} |" + "{:>4}"*6)
        frmt2 = ("{:>4}"*4 + " "*5 + "{:>4}"*4)
        frmt3 = ("{:>6}" + " "*5 + "{:>6}")

        tadReward, KthReward, susyReward = self.fitnessDetails[:3]
        tadpoles, fillers, Kth           = self.fitnessDetails[3:6]
        uBest, Xterms, Yterms            = self.fitnessDetails[6:]

        argsTad = ["tadpole", self.weights[0]*tadReward, self.weights[0], tadReward]
        argsKth = ["K-theory", self.weights[1]*KthReward, self.weights[1], KthReward]
        argsSUSY = ["SUSY", self.weights[2]*susyReward, self.weights[2], susyReward]
        frmtTad = ("\n{:>10} = {:.4f}\t= {:.2f} x {:.4f}")
        frmtKth = ("{:>10} = {:.4f}\t= {:.2f} x {:.2f}")
        frmtSUSY = ("{:>10} = {:.4f}\t= {:.2f} x {:.4f}")
        frmtXY = ("{:7.4f}" + " "*5 + "{:7.4f}")

        print("\n", frmt1.format(*hdr1), end=' '*5)
        print(frmt2.format(*hdr2), end=' '*5)
        print(frmt3.format(*hdr3))
        print('-'*98)

        for stack, i in zip(self.stacks, range(self.numStacks)):
            print(frmt1.format(*stack), end=' '*5)
            print(frmt2.format(*getStackXY(stack, self.bix2)), end=' '*5)
            print(frmtXY.format(Xterms[i], Yterms[i]))

        print(frmtTad.format(*argsTad), end='\t\t')
        print(("{:>4}"*4).format(*tadpoles), "\t" + ("{:>4.0f}"*4).format(*fillers))
        print(frmtKth.format(*argsKth), end='\t\t')
        print(("{:>4}"*4).format(*Kth))
        print(frmtSUSY.format(*argsSUSY), end='\t\t')
        print(("{:8.4f}"*4).format(*uBest), end='\n')
        print(("{:>10} = {:.4f}").format(*["fitness", self.fitness]), end='\t\t\t\t')
        print(("{:8.2f}"*4).format(*uBest/min(np.abs(uBest))), end='\n\n')


def randomStack():
    s = np.zeros(7, dtype='int')
    s[0] = np.random.geometric(0.5)
    s[1:] = np.random.binomial(2*20, 0.5, size=6) - 20

    return s


def randomBnml(cap, n):
    return np.random.binomial(2*cap, 0.5, size=n) - cap


def getStackXY(stack, bix2):
    n1, m1, n2, m2, n3, m3 = stack[1:]
    b1x2, b2x2, b3x2 = bix2

    m1hat = (1 + b1x2) * m1 + b1x2 * n1
    m2hat = (1 + b2x2) * m2 + b2x2 * n2
    m3hat = (1 + b3x2) * m3 + b3x2 * n3

    X0 = + n1 * n2 * n3
    X1 = - n1 * m2hat * m3hat
    X2 = - m1hat * n2 * m3hat
    X3 = - m1hat * m2hat * n3

    Y0 = + m1hat * m2hat * m3hat
    Y1 = - m1hat * n2 * n3
    Y2 = - n1 * m2hat * n3
    Y3 = - n1 * n2 * m3hat

    return X0, X1, X2, X3, Y0, Y1, Y2, Y3


def SUSY(Ns, XYlist):
    # The SUSY (in)equalities are overconstrained. Find solution
    # for a subset of stacks and see if it extends to the rest.

    eps = 10**(-4)


    Xlist = XYlist[:, :4]
    Ylist = XYlist[:, 4:]

    # for each stack find how many YI are zero
    zeroYcounts = np.sum(Ylist == 0, axis=1)

    # # filler branes have all YI=0 and the SUSY Y-term is automatically solved
    # fillers = np.where(zeroYcounts == 4)[0]

    # # stacks with exactly one nonzero YI cannot possibly satisfy
    # # the SUSY Y-term. These should be penalized.
    # oneNonzeroY = np.where(zeroYcounts == 3)[0]

    # the rest: two or more YI are nonzero
    others = np.where((zeroYcounts <= 2) * (Ns > 0))[0]

    susyBest = np.inf
    uBest = [1, 1, 1, 1]
    XtermsBest = np.inf * np.ones(len(Ns))
    YtermsBest = np.inf * np.ones(len(Ns))

    if len(others) >= 3:

        for i in range(5):
            inds = np.random.choice(others, size=3, replace=False)
            matrix = Ylist[inds, 1:]

            if np.linalg.det(matrix) != 0:
                uiInv = np.dot(np.linalg.inv(matrix), -Ylist[inds, 0])

                if min(uiInv) < eps or max(uiInv) > 1/eps:
                    # ratios of moduli must not be too large or too small
                    continue

                uI = np.block([1, 1/uiInv])
                uI /= np.sqrt(uI@uI)

                Xterms = np.array([np.minimum(0, np.sign(Ns[a]) * Xlist[a] @ uI)
                                   for a in range(len(Xlist))])

                Yterms = np.array([np.sign(Ns[a]) * Ylist[a] @ (1/uI) for a in range(len(Ylist))])

                val = Yterms@Yterms + Xterms@Xterms

                if val < susyBest:
                    susyBest = val
                    uBest = uI
                    XtermsBest = Xterms
                    YtermsBest = Yterms


    if susyBest == np.inf:
        # optimize
        
        bounds = [(eps, 1), (eps, 1), (eps, 1), (eps, 1)]
        # cons = ({'type': 'eq', 'fun': S3})

        # result = minimize(SUSYfunc, x0=(0.5, 0.5, 0.5, 0.5), args=(Ns, Xlist, Ylist),
        #                   method='SLSQP', bounds=bounds, constraints=cons)

        # susyBest = result.fun
        # uBest = result.x

        # Xterms = np.array([np.minimum(0, np.sign(Ns[a]) * Xlist[a] @ uBest)
        #                    for a in range(len(Xlist))])

        # Yterms = np.array([np.sign(Ns[a]) * Ylist[a] @ (1/uBest) for a in range(len(Ylist))])


    return susyBest, uBest, XtermsBest, YtermsBest


def S3(x):
    return x@x - 1

def SUSYfunc(x, Ns, Xlist, Ylist):

    Xterms = np.array([np.minimum(0, np.sign(Ns[a]) * Xlist[a] @ x) for a in range(len(Xlist))])
    Yterms = np.array([np.sign(Ns[a]) * Ylist[a] @ (1/x) for a in range(len(Ylist))])

    return Xterms@Xterms + Yterms@Yterms


def MSSM(Ns, Xlist, Ylist):

    # filler branes have all YI=0
    filler = [YI@YI == 0 for YI in Ylist]

    # SU(3) factors
    SU3inds = np.where((Ns == 3) * (not filler))[0]

    # SU(2) factors
    SU2inds = np.where((Ns == 2) * (not filler))[0]

    # USp(1) (=SU(2)) factors
    USp1inds = np.where((Ns == 1) * filler)[0]

    gaugeGroupDist = max(0, 1 - len(SU3inds)) + max(0, 1 - len(SU2inds) - len(USp1inds))

    if gaugeGroupDist == 0:
        # contains SU(3) x SU(2) x U(1)

        # loop over ways to pick which factors correspond to SM gauge group
        for i3 in SU3inds:
            for i2 in np.block([SU2inds, USp1inds]):
                1


    MSSMbonus = 0.25 * np.exp(-gaugeGroupDist)


    return MSSMbonus
