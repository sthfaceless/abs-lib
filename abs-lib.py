from enum import Enum
import numpy as np
import math
from cvxopt import matrix, solvers


class KnowledgePatternManager:
    def checkInconsistency(self, knowledgePattern):
        return self.__getInconsistencyChecker(knowledgePattern.getType()) \
            .isInconsistent(knowledgePattern)

    def __getInconsistencyChecker(self, type):  # его я не трогаю.
        if type == KnowledgePatternType.QUANTS:
            return QuantInconsistencyChecker()
        elif type == KnowledgePatternType.DISJUNCTS:
            return DisjunctInconsistencyChecker()
        elif type == KnowledgePatternType.CONJUNCTS:
            return ConjunctInconsistencyChecker()
        else:
            raise TypeError("Incorrect type of knowledge pattern")  # ???


class KnowledgePatternType(Enum):
    QUANTS = 'quants',
    DISJUNCTS = 'disjuncts',
    CONJUNCTS = 'conjuncts'


class InconsistencyChecker:
    @staticmethod                             #нужен ли
    def isInconsistent(knowledgePattern):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")


class QuantInconsistencyChecker(InconsistencyChecker):
    @staticmethod
    def isInconsistent(knowledgePattern):
        size = knowledgePattern.size
        quantMatrix = MatrixProducer.getIdentityMatrix(size)
        intervalsArray = knowledgePattern.array
        if LinearProgrammingProblemSolver.findOptimalValues(quantMatrix, intervalsArray,size).isInconsistent == False:  # тут не то мб
            return InconsistencyResult(False, [])
        else:
            array = LinearProgrammingProblemSolver.findOptimalValues(quantMatrix, intervalsArray, size).array
            return LinearProgrammingProblemSolver.findNormalizedOptimalValues(array, size)


class ConjunctInconsistencyChecker(InconsistencyChecker):
    @staticmethod
    def isInconsistent(knowledgePattern):
        size = knowledgePattern.size
        conjunctMatrix = MatrixProducer.getConjunctToQuantMatrix(size)
        intervalsArray = knowledgePattern.array
        return LinearProgrammingProblemSolver.findOptimalValues(conjunctMatrix, intervalsArray, size)


class DisjunctInconsistencyChecker(InconsistencyChecker):
    @staticmethod
    def isInconsistent(knowledgePattern):
        size = knowledgePattern.size
        disjunctMatrix = MatrixProducer.getDisjunctToQuantMatrix(size)
        intervalsArray = knowledgePattern.array()
        return LinearProgrammingProblemSolver.findOptimalValues(disjunctMatrix, intervalsArray, size)


class MatrixProducer:
    @staticmethod
    def getDisjunctToQuantMatrix(size):
        return np.linalg.inv(MatrixProducer.getQuantToDisjunctMatrix(math.log(size, 2)))

    @staticmethod
    def getQuantToDisjunctMatrix(n):
        if n == 0:
            return np.array([1], dtype=np.double)
        elif n == 1:
            return np.array([[1, 1], [0, 1]], dtype=np.double)
        else:
            k = MatrixProducer.getQuantToDisjunctMatrix(n - 1)
            i = np.ones((2 ** (n - 1), 2 ** (n - 1)), dtype=np.double)
            k_o = k.copy()
            k_o[0] = [0] * 2 ** (n - 1)
            return np.block([[k, k], [k_o, i]])

    @staticmethod
    def getConjunctToQuantMatrix(n):
        if n == 0:
            return np.array([1], dtype=np.double)
        elif n == 1:
            return np.array([[1, -1], [0, 1]], dtype=np.double)
        else:
            i = MatrixProducer.getConjunctToQuantMatrix(n - 1)
            o = np.zeros((2 ** (n - 1), 2 ** (n - 1)), dtype=np.double)
            return np.block([[i, (-1) * i], [o, i]])

    @staticmethod
    def getIdentityMatrix(size):
        return np.eye(size, dtype=np.double)


class LinearProgrammingProblemSolver:

    @staticmethod
    def findOptimalValues(matrixs, array, size):
        a = np.vstack(((-1) * matrixs, (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        a = matrix(a)
        b = np.hstack((np.zeros(size, dtype=np.double), (-1) * array[:, 0], array[:, 1]))
        b = matrix(b)
        c = np.array(np.zeros(size, dtype=np.double))
        c = matrix(c)
        solvers.options['show_progress'] = False
        valid = True
        resultArray = array.copy()
        for i in range(size):
            c[i] = 1
            sol = solvers.lp(c, a, b)
            if sol['status'] == 'Optimal':  # неуверенность
                valid = False
                resultArray = []
                break
            resultArray[i][0] = round(sol['x'][i], 3)
            c[i] = -1
            sol = solvers.lp(c, a, b)
            if sol['status'] == 'Optimal':
                valid = False
                resultArray = []
                break
            resultArray[i][1] = round(sol['x'][i], 3)
            c[i] = 0
        return InconsistencyResult(valid, resultArray)

    @staticmethod
    def findNormalizedOptimalValues(array, size):
        a = np.vstack(((-1) * np.ones(size, dtype=np.double), np.ones(size, dtype=np.double),
                       (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        a = matrix(a)
        b = np.hstack(
            ((-1) * np.ones(1, dtype=np.double), np.ones(1, dtype=np.double), (-1) * array[:, 0], array[:, 1]))
        b = matrix(b)
        c = np.array(np.zeros(size, dtype=np.double))
        c = matrix(c)
        valid = True
        resultArray = array.copy()
        for i in range(size):
            c[i] = 1
            sol = solvers.lp(c, a, b)
            if sol['status'] == 'Optimal':
                valid = False
                resultArray = []
                break
            resultArray[i][0] = round(sol['x'][i], 3)
            c[i] = -1
            sol = solvers.lp(c, a, b)
            if sol['status'] == 'Optimal':
                valid = False
                resultArray = []
                break
            resultArray[i][1] = round(sol['x'][i], 3)
            c[i] = 0
        return InconsistencyResult(valid, resultArray)


class InconsistencyResult:  # &&&??
    def __init__(self, inconsistent, arr):
        self.inconsistent = inconsistent
        self.arr = arr

    @property                            #неясно
    def array(self):
        if self.inconsistent != True:
            raise AttributeError('There is no have array, because knowledge pattern is inconsistency')
        else:
            return self.arr

    @property
    def isInconsistent(self):
        return self.inconsistent


class KnowledgePatternItem:
    def __init__(self, arr, type):
        self.type = type
        self.arr = arr

    @property
    def type(self):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")

    def getElement(self, index):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")

    @property
    def array(self):
        return NotImplementedError("It's a method of abstract class, use appropriate implementation")

    @property
    def size(self):
        return NotImplementedError("It's a method of abstract class, use appropriate implementation")


class QuantKnowledgePatternItem(KnowledgePatternItem):

    @property
    def type(self):
        return self.type

    def getElement(self, index):
        return self.arr[index]

    @property
    def array(self):
        return self.arr

    @property
    def size(self):
        return len(self.arr)


class DisjunctKnowledgePatternItem(KnowledgePatternItem):
    @property
    def type(self):
        return self.type

    def getElement(self, index):
        return self.arr[index]

    @property
    def array(self):
        return self.arr

    @property
    def size(self):
        return len(self.arr)


class ConjunctKnowledgePatternItem(KnowledgePatternItem):
    @property
    def type(self):
        return self.type

    def getElement(self, index):
        return self.arr[index]

    @property
    def array(self):
        return self.arr

    @property
    def size(self):
        return len(self.arr)


