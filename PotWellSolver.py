from operator import itemgetter
from pylab import *
from scipy.sparse import diags
from scipy.sparse import bmat
from scipy.sparse.linalg import eigsh
massElectron = 5.6778*10**(-13)  # in meV.s^2/cm^2
hbar = 6.58211928*10**(-13)  # in meV.s
wellWidth = 100*10**(-8)  # in cm
pi = math.pi
unitE = (hbar**2*pi**2)/(2*massElectron*wellWidth**2)

class PotWellSolver:

    def __init__(self, compound, potWell, matrixDim=4):
        self.compound = compound
        self.potWell = potWell
        self.matrixDim = matrixDim
        self.unitV = self.potWell.depth/unitE

        self.unitDelta = self.compound.delta/unitE

        self.Dense = 1
        self.nGridPoints = 100
        self.xMax = 3
        self.xMin = -3
        self.xAxisVector = np.linspace(self.xMin, self.xMax, self.nGridPoints)
        self.stepSize = (self.xMax-self.xMin)/float(len(self.xAxisVector))
        self.potWellBoundary1 = np.floor(len(self.xAxisVector)/2) - np.floor(self.potWell.width/(2.*self.stepSize))

        self.potWellBoundary2 = np.floor(len(self.xAxisVector)/2) + np.ceil(self.potWell.width/(2.*self.stepSize))

        self.potWellCenter = np.floor(len(self.xAxisVector)/2)


    def setParameters(self, nGridPoints, xMin=-3, xMax=3):
        self.nGridPoints = nGridPoints
        self.xMin = xMin
        self.xMax = xMax
        self.xAxisVector = np.linspace(self.xMin, self.xMax, self.nGridPoints)
        self.stepSize = (self.xMax-self.xMin)/float(len(self.xAxisVector))
        self.potWellBoundary1 = np.floor(len(self.xAxisVector)/2) - np.floor(self.potWell.width/(2.*self.stepSize))
        self.potWellBoundary2 = np.floor(len(self.xAxisVector)/2) + np.ceil(self.potWell.width/(2.*self.stepSize))
        self.potWellCenter = np.floor(len(self.xAxisVector)/2)

    def setDense(self, bDense):
        self.Dense = bDense

    def getXAxisVector(self):
        return self.xAxisVector

    def setGridPoints(self, nGridPoints):
        self.setParameters(nGridPoints)

    def makeMatrix(self, k, BULK=False):
        ky = 0

        P = None
        Q = None
        R= None
        S = None
        V = None
        HKL = None

        diagP = 0
        subdiagP = 0
        superdiagP = 0
        diagQ = 0
        subdiagQ = 0
        superdiagQ = 0
        diagS = 0
        subdiagS = 0
        superdiagS = 0
        diagR = 0
        subdiagR = 0
        superdiagR = 0
        if not BULK:
            if self.potWell.nDirection == 1:
                diagP = (self.compound.y1/pi**2)*(k**2 + ky**2 + 2./self.stepSize**2)
                subdiagP = -self.compound.y1/(self.stepSize**2*pi**2)
                superdiagP = subdiagP


                diagQ = (self.compound.y2/pi**2) * ((ky**2-2*k**2)+(2/self.stepSize**2))
                subdiagQ = -self.compound.y2/(self.stepSize**2*pi**2)
                superdiagQ = subdiagQ



                diagR = (np.sqrt(3)*self.compound.y2/pi**2) * (-2/self.stepSize**2 + ky**2)
                subdiagR = (np.sqrt(3)/pi**2)*(self.compound.y2/self.stepSize**2 - self.compound.y3*ky/self.stepSize)
                superdiagR = (np.sqrt(3)/pi**2)*(self.compound.y2/self.stepSize**2 + self.compound.y3*ky/self.stepSize)


                diagS = -2.*self.compound.y3*np.sqrt(3)*1j*ky*k/pi**2
                subdiagS = np.sqrt(3)*self.compound.y3*k*1j/(pi**2*self.stepSize)
                superdiagS = -subdiagS



            elif self.potWell.nDirection == 3:
                diagP = (self.compound.y1/pi**2)*(k**2 + ky**2 + 2./self.stepSize**2)

                subdiagP = -self.compound.y1/(self.stepSize**2*pi**2)

                superdiagP = subdiagP


                diagQ = (self.compound.y2/pi**2) * (k**2 + ky**2 - (4/self.stepSize**2))

                subdiagQ = 2*self.compound.y2/(self.stepSize**2*pi**2)

                superdiagQ = subdiagQ


                diagR = (1/pi**2) * (-np.sqrt(3)*self.compound.y2*(k**2 - ky**2) + 1j*2*np.sqrt(3)*self.compound.y3*k*ky)


                subdiagS = (1j*self.compound.y3*np.sqrt(3))/(pi**2*self.stepSize) * (k-1j*ky)

                superdiagS = -subdiagS




            if self.Dense:
                P = zeros((self.nGridPoints, self.nGridPoints), dtype=complex)
                Q = zeros((self.nGridPoints, self.nGridPoints), dtype=complex)
                R = zeros((self.nGridPoints, self.nGridPoints), dtype=complex)
                S = zeros((self.nGridPoints, self.nGridPoints), dtype=complex)
                i, j = indices(P.shape)
                P[i == j] = diagP
                P[i == j-1] = superdiagP
                P[i == j+1] = subdiagP

                i, j = indices(Q.shape)
                Q[i == j] = diagQ
                Q[i == j-1] = superdiagQ
                Q[i == j+1] = subdiagQ

                i, j = indices(R.shape)
                R[i == j] = diagR
                R[i == j-1] = superdiagR
                R[i == j+1] = subdiagR

                i, j = indices(S.shape)
                S[i == j] = diagS
                S[i == j-1] = superdiagS
                S[i == j+1] = subdiagS

                potVec = zeros(self.nGridPoints)
                potVec[0:self.potWellBoundary1] = self.unitV
                potVec[self.potWellBoundary2:self.nGridPoints] = self.unitV
                V = diag(potVec)

            elif not self.Dense:
                Pdiag = ones(self.nGridPoints)*diagP
                Psubdiag = ones(self.nGridPoints-1)*subdiagP
                Psuperdiag = ones(self.nGridPoints-1)*superdiagP
                P = diags([Psubdiag, Pdiag, Psuperdiag], [-1, 0, 1], format="csc")

                Qdiag = ones(self.nGridPoints)*diagQ
                Qsubdiag = ones(self.nGridPoints-1)*subdiagQ
                Qsuperdiag = ones(self.nGridPoints-1)*superdiagQ
                Q = diags([Qsubdiag, Qdiag, Qsuperdiag], [-1, 0, 1], format="csc")

                Rdiag = ones(self.nGridPoints)*diagR
                Rsubdiag = ones(self.nGridPoints-1)*subdiagR
                Rsuperdiag = ones(self.nGridPoints-1)*superdiagR
                R = diags([Rsubdiag, Rdiag, Rsuperdiag], [-1, 0, 1], format="csc")

                Sdiag = ones(self.nGridPoints)*diagS
                Ssubdiag = ones(self.nGridPoints-1)*subdiagS
                Ssuperdiag = ones(self.nGridPoints-1)*superdiagS
                S = diags([Ssubdiag, Sdiag, Ssuperdiag], [-1, 0, 1], format="csc")

                potVec = zeros(self.nGridPoints)
                potVec[0:self.potWellBoundary1] = self.unitV
                potVec[self.potWellBoundary2:self.nGridPoints] = self.unitV
                V = diags([potVec], [0], format="csc")
        elif BULK:
            fraction = hbar**2/(2*massElectron)
            P = self.compound.y1*k**2*fraction
            Q = -2*self.compound.y2*k**2*fraction
            R = 0
            S = 0

            if self.matrixDim == 6:
                Delta = self.compound.delta
                HKL = -np.matrix([[P+Q, -S, R, 0, -S/np.sqrt(2), np.sqrt(2)*R],
                        [-S.conjugate(), P-Q, 0, R, -np.sqrt(2)*Q, np.sqrt(3./2.)*S],
                        [R.conjugate(), 0, P-Q, S, np.sqrt(3./2.)*S.conjugate(), np.sqrt(2)*Q],
                        [0, R.conjugate(), S.conjugate(), P+Q, -np.sqrt(2)*R.conjugate(), -S.conjugate()/np.sqrt(2)],
                        [-S.conjugate()/np.sqrt(2), -np.sqrt(2)*Q.conjugate(), np.sqrt(3./2.)*S, -np.sqrt(2)*R, P+Delta, 0],
                        [np.sqrt(2)*R.conjugate(), np.sqrt(3./2.)*S.conjugate(), np.sqrt(2)*Q.conjugate(), -S/np.sqrt(2), 0, P+Delta]])
            elif self.matrixDim == 4:
                HKL = -np.matrix([[P+Q, -S, R, 0],
                                  [-S.conjugate(), P-Q, 0, R],
                                  [R.conjugate(), 0, P-Q, S],
                                  [0, R.conjugate(), S.conjugate(), P+Q]])




        if not BULK:
            if self.Dense:
                if self.matrixDim == 6:
                    Delta = diag((ones(self.nGridPoints)*self.unitDelta))
                    HKL = np.bmat([[P+Q+V, -S, R, zeros((self.nGridPoints, self.nGridPoints)), -S/np.sqrt(2), np.sqrt(2)*R],
                                [-S.conj().T, P-Q+V, zeros((self.nGridPoints, self.nGridPoints)), R, -np.sqrt(2)*Q, np.sqrt(3./2.)*S],
                                [R.conj().T, zeros((self.nGridPoints, self.nGridPoints)), P-Q+V, S, np.sqrt(3./2.)*S.conj().T, np.sqrt(2)*Q],
                                [zeros((self.nGridPoints, self.nGridPoints)), R.conj().T, S.conj().T, P+Q+V, -np.sqrt(2)*R.conj().T, -S.conj().T/np.sqrt(2)],
                                [-S.conj().T/np.sqrt(2), -np.sqrt(2)*Q.conj().T, np.sqrt(3./2.)*S, -np.sqrt(2)*R, P+Delta+V, zeros((self.nGridPoints, self.nGridPoints))],
                                [np.sqrt(2)*R.conj().T, np.sqrt(3./2.)*S.conj().T, np.sqrt(2)*Q.conj().T, -S/np.sqrt(2), zeros((self.nGridPoints, self.nGridPoints)), P+Delta+V]])
                elif self.matrixDim == 4:
                    HKL = np.bmat([[P+Q+V, -S, R, zeros((self.nGridPoints, self.nGridPoints))],
                                [-S.conj().T, P-Q+V, zeros((self.nGridPoints, self.nGridPoints)), R],
                                [R.conj().T, zeros((self.nGridPoints, self.nGridPoints)), P-Q+V, S],
                                [zeros((self.nGridPoints, self.nGridPoints)), R.conj().T, S.conj().T, P+Q+V]])
            elif not self.Dense:
                if self.matrixDim == 6:
                    Delta = diags([ones(self.nGridPoints)*self.unitDelta], [0])
                    HKL = bmat([[P+Q+V, -S, R, zeros((self.nGridPoints, self.nGridPoints)), -S/np.sqrt(2), np.sqrt(2)*R],
                                [-S.conj().T, P-Q+V, zeros((self.nGridPoints, self.nGridPoints)), R, -np.sqrt(2)*Q, np.sqrt(3./2.)*S],
                                [R.conj().T, zeros((self.nGridPoints, self.nGridPoints)), P-Q+V, S, np.sqrt(3./2.)*S.conj().T, np.sqrt(2)*Q],
                                [zeros((self.nGridPoints, self.nGridPoints)), R.conj().T, S.conj().T, P+Q+V, -np.sqrt(2)*R.conj().T, -S.conj().T/np.sqrt(2)],
                                [-S.conj().T/np.sqrt(2), -np.sqrt(2)*Q.conj().T, np.sqrt(3./2.)*S, -np.sqrt(2)*R, P+Delta+V, zeros((self.nGridPoints, self.nGridPoints))],
                                [np.sqrt(2)*R.conj().T, np.sqrt(3./2.)*S.conj().T, np.sqrt(2)*Q.conj().T, -S/np.sqrt(2), zeros((self.nGridPoints, self.nGridPoints)), P+Delta+V]], format="csc")
                elif self.matrixDim == 4:
                    HKL = bmat([[P+Q+V, -S, R, zeros((self.nGridPoints, self.nGridPoints))],
                                [-S.conj().T, P-Q+V, zeros((self.nGridPoints, self.nGridPoints)), R],
                                [R.conj().T, zeros((self.nGridPoints, self.nGridPoints)), P-Q+V, S],
                                [zeros((self.nGridPoints, self.nGridPoints)), R.conj().T, S.conj().T, P+Q+V]], format="csc")
        return HKL


    def calcEigs(self, k, nSmallest=6, BULK=False):
        HKL = self.makeMatrix(k, BULK)
        w = None
        v = None
        if self.Dense:
            w, v = eigh(HKL)
        elif not self.Dense:
            w, v = eigsh(HKL, nSmallest, None, None, "SM")
        return w*unitE, v

    def calcEigVals(self, k, nSmallest=6, BULK=False):
        HKL = self.makeMatrix(k,BULK)
        w = None
        if self.Dense:
            w = eigvalsh(HKL)
        elif not self.Dense:
            w,v = eigsh(HKL, nSmallest, None, None, "SM")
        if not BULK:
            return w*unitE
        elif BULK:
            return w

    def getEigenValues(self, kVec, nSmallest, BULK=False):
        if not BULK:
            if type(kVec) == int:
                eigenValues = self.calcEigVals(kVec)
                eigenValues = sorted(eigenValues)
                EArray = np.zeros(nSmallest)
                for i in xrange(0, nSmallest):
                    EArray[i] = eigenValues[i*2].real
                return EArray
            else:
                EMatrix = np.zeros((nSmallest, len(kVec)))
                column = 0
                for k in kVec:
                    eigenValues = self.calcEigVals(k, nSmallest*2)
                    eigenValues = sorted(eigenValues)
                    for i in xrange(0, nSmallest):
                        EMatrix[i][column] = eigenValues[i*2].real
                    column += 1
                return EMatrix
        elif BULK:
            EMatrix = zeros((nSmallest/2, len(kVec)))
            column = 0
            for k in kVec:
                eigenValues = self.calcEigVals(k, nSmallest, BULK)
                eigenValues = sorted(eigenValues)
                for i in xrange(nSmallest/2):
                    EMatrix[i,column] = eigenValues[i*2]
                column += 1
            return EMatrix

    def getEigenvectors(self, k, state=0, BULK=False):
        w, v = self.calcEigs(k,BULK)
        data = [(w[i], v[:,i]) for i in xrange(0, self.matrixDim*self.nGridPoints)]
        data = sorted(data, key=itemgetter(0))
        return data[state][1]

    def getMixing(self, k, state=0, BULK=False):
        w, v = self.calcEigs(k,BULK)
        data = [(w[i], v[:,i]) for i in xrange(self.matrixDim*self.nGridPoints)]
        data = sorted(data, key=itemgetter(0))
        eigenVector = data[state][1]
        splitVectors = np.zeros((self.nGridPoints, self.matrixDim), dtype=complex)
        for i in xrange(self.matrixDim):
            splitVectors[:,i] = np.squeeze(np.array(eigenVector[i*self.nGridPoints:(i+1)*self.nGridPoints]))

        normSQ = np.zeros(self.matrixDim)
        for i in xrange(self.matrixDim):
            normSQ[i] = norm(splitVectors[:,i])**2

        totalDensity = sum(normSQ)

        fractions = []
        for i in xrange(self.matrixDim):
            fractions.append(normSQ[i]/totalDensity)
        return fractions


    def rotateMixing(self, k, rotateTo="z", State=0):
        eigenVectors = self.getEigenvectors(k, state=State)

        splitVectors = np.zeros((self.nGridPoints, self.matrixDim), dtype=complex)
        for i in xrange(self.matrixDim):
            splitVectors[:,i] = np.squeeze(np.array(eigenVectors[i*self.nGridPoints:(i+1)*self.nGridPoints]))


        H1rot = None
        L1rot = None
        L2rot = None
        H2rot = None

        if rotateTo == "z":
            H1rot = np.sqrt(2)/4 * (splitVectors[:,0] + splitVectors[:,3]) + np.sqrt(6)/4 * (splitVectors[:,1] + splitVectors[:,2])
            L1rot = np.sqrt(6)/4 * (splitVectors[:,3] - splitVectors[:,0]) + np.sqrt(2)/4 * (-splitVectors[:,1] + splitVectors[:,2])
            L2rot = np.sqrt(6)/4 * (splitVectors[:,0] + splitVectors[:,3]) + np.sqrt(2)/4 * (-splitVectors[:,1] - splitVectors[:,2])
            H2rot = np.sqrt(2)/4 * (splitVectors[:,3] - splitVectors[:,0]) + np.sqrt(6)/4 * (splitVectors[:,1] - splitVectors[:,2])
        elif rotateTo == "x":
            H1rot = np.sqrt(2)/4 * (splitVectors[:,0] - splitVectors[:,3]) + np.sqrt(6)/4 * (splitVectors[:,2] - splitVectors[:,1])
            L1rot = np.sqrt(6)/4 * (splitVectors[:,3] + splitVectors[:,0]) + np.sqrt(2)/4 * (-splitVectors[:,1] - splitVectors[:,2])
            L2rot = np.sqrt(6)/4 * (splitVectors[:,0] - splitVectors[:,3]) + np.sqrt(2)/4 * (splitVectors[:,1] - splitVectors[:,2])
            H2rot = np.sqrt(2)/4 * (splitVectors[:,3] + splitVectors[:,0]) + np.sqrt(6)/4 * (splitVectors[:,1] + splitVectors[:,2])


        HH1 = norm(H1rot)**2
        HH2 = norm(H2rot)**2
        LH1 = norm(L1rot)**2
        LH2 = norm(L2rot)**2
        Total = HH1 + HH2 + LH1 + LH2
        LH1Frac = LH1/Total
        LH2Frac = LH2/Total


        HHTotalFrac = (HH1 + HH2)/Total
        LHTotalFrac = LH1Frac + LH2Frac

        both = []
        both.append(HHTotalFrac)
        both.append(LHTotalFrac)
        return both


    def setXMax(self, xMax):
        self.setParameters(self.nGridPoints, self.xMin, xMax)

    def setXMin(self, xMin):
        self.setParameters(self.nGridPoints, xMin, self.xMax)

    def setXRange(self, xMin, xMax):
        self.setParameters(self.nGridPoints, xMin, xMax)






