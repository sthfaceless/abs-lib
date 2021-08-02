import math
from enum import Enum

import numpy as np
from cvxopt import matrix, solvers


class KnowledgePatternManager:
    @staticmethod
    def checkConsistency(knowledgePattern):
        return KnowledgePatternManager.__getConsistencyChecker(knowledgePattern.type) \
            .isConsistent(knowledgePattern)

    @staticmethod
    def __getConsistencyChecker(type):
        if type == KnowledgePatternType.QUANTS:
            return QuantConsistencyChecker()
        elif type == KnowledgePatternType.DISJUNCTS:
            return DisjunctConsistencyChecker()
        elif type == KnowledgePatternType.CONJUNCTS:
            return ConjunctConsistencyChecker()
        else:
            raise TypeError("Correct type of knowledge pattern")


class KnowledgePatternType(Enum):
    QUANTS = 'quants',
    DISJUNCTS = 'disjuncts',
    CONJUNCTS = 'conjuncts'


class ConsistencyChecker:
    @staticmethod
    def isConsistent(knowledgePattern):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")


class QuantConsistencyChecker(ConsistencyChecker):
    @staticmethod
    def isConsistent(knowledgePattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getIdentityMatrix(size)
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        result = LinearProgrammingProblemSolver.findOptimalValues(matrix, intervals, size)
        if result.consistent:
            result = LinearProgrammingProblemSolver.findNormalizedOptimalValues(np.array(result.array, dtype=np.double),
                                                                                size)
        return result


class ConjunctConsistencyChecker(ConsistencyChecker):
    @staticmethod
    def isConsistent(knowledgePattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getConjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        return LinearProgrammingProblemSolver.findOptimalValues(matrix, intervals, size)


class DisjunctConsistencyChecker(ConsistencyChecker):
    @staticmethod
    def isConsistent(knowledgePattern):
        size = knowledgePattern.size
        matrix = MatrixProducer.getDisjunctsToQuantsMatrix(int(math.log(size, 2)))
        intervals = np.array(knowledgePattern.array, dtype=np.double)
        return LinearProgrammingProblemSolver.findOptimalValues(matrix, intervals, size)


class MatrixProducer:
    @staticmethod
    def getDisjunctsToQuantsMatrix(n):
        return np.linalg.inv(MatrixProducer.getQuantsToDisjunctsMatrix(n))

    @staticmethod
    def getQuantsToDisjunctsMatrix(n):
        if n == 0:
            return np.array([1], dtype=np.double)
        elif n == 1:
            return np.array([[1, 1], [0, 1]], dtype=np.double)
        else:
            k = MatrixProducer.getQuantsToDisjunctsMatrix(n - 1)
            i = np.ones((2 ** (n - 1), 2 ** (n - 1)), dtype=np.double)
            k_o = k.copy()
            k_o[0] = [0] * 2 ** (n - 1)
            return np.block([[k, k], [k_o, i]])

    @staticmethod
    def getConjunctsToQuantsMatrix(n):
        if n == 0:
            return np.array([1], dtype=np.double)
        elif n == 1:
            return np.array([[1, -1], [0, 1]], dtype=np.double)
        else:
            i = MatrixProducer.getConjunctsToQuantsMatrix(n - 1)
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
        return LinearProgrammingProblemSolver.optimizeForMatrices(a, b, c, size, array)

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
        return LinearProgrammingProblemSolver.optimizeForMatrices(a, b, c, size, array)

    @staticmethod
    def optimizeForMatrices(a, b, c, size, intervals):
        solvers.options['show_progress'] = False
        _intervals = intervals.copy()
        for i in range(size):
            c[i] = 1
            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return ConsistencyResult(False, [])
            _intervals[i][0] = round(sol['x'][i], 3)

            c[i] = -1
            sol = solvers.lp(c, a, b)
            if sol['status'] != 'optimal':
                return ConsistencyResult(False, [])
            _intervals[i][1] = round(sol['x'][i], 3)
            c[i] = 0
        return ConsistencyResult(True, _intervals.tolist())


class ConsistencyResult:
    def __init__(self, consistent, arr):
        self._consistent = consistent
        self._arr = arr

    @property
    def array(self):
        if self._consistent:
            return self._arr
        else:
            raise AttributeError('There is no have array, because knowledge pattern is Consistency')

    @property
    def consistent(self):
        return self._consistent


class KnowledgePatternItem:
    def __init__(self, array):
        self._arr = array

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
    _type = KnowledgePatternType.QUANTS

    @property
    def type(self):
        return self._type

    def getElement(self, index):
        return self._arr[index]

    @property
    def array(self):
        return self._arr

    @property
    def size(self):
        return len(self._arr)


class DisjunctKnowledgePatternItem(KnowledgePatternItem):
    _type = KnowledgePatternType.DISJUNCTS

    @property
    def type(self):
        return self._type

    def getElement(self, index):
        return self._arr[index]

    @property
    def array(self):
        return self._arr

    @property
    def size(self):
        return len(self._arr)


class ConjunctKnowledgePatternItem(KnowledgePatternItem):
    _type = KnowledgePatternType.CONJUNCTS

    @property
    def type(self):
        return self._type

    def getElement(self, index):
        return self._arr[index]

    @property
    def array(self):
        return self._arr

    @property
    def size(self):
        return len(self._arr)
