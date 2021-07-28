from enum import Enum
import numpy as np
import math
from cvxopt import matrix, solvers # переменные с маленькой буквы -> ссылки на обьекты

class KnowledgePatternManager:
    def checkInconsistency(self, knowledgePattern):                          #у этого объекта есть метод, который получает объект KnowledgePatternItem
        return self.__getInconsistencyChecker(knowledgePattern.getType()) \
            .isInconsistent(knowledgePattern)                                # он возвращает резульат метода класса QuantInconsistencyChecker

    def __getInconsistencyChecker(self, type):                               # этому методу нужен тип как объект класса, возварщает он объект указанного класса
        if type == KnowledgePatternType.QUANTS:                               # KnowledgePattern охарактеризовывается типом, все ок, то есть type, это та самая штука охарактеризовывающая
            return QuantInconsistencyChecker()
        elif type == KnowledgePatternType.DISJUNCTS:
            return DisjunctInconsistencyChecker()
        else:
            return ConjuctInconsistencyChecker()


class KnowledgePatternType(Enum):
    QUANTS = 'quants',
    DISJUNCTS = 'disjuncts',
    CONJUCTS = 'conjucts'


class InconsistencyChecker:
    def isInconsistent(self, knowledgePattern):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")


class QuantInconsistencyChecker(InconsistencyChecker):
    def isInconsistent(self, knowledgePattern):
        size = knowledgePattern.getSize()
        QuantMatrix = MatrixProducer.getQuantMatrix(size)
        IntervalsArray = knowledgePattern.getArray()
        if LinearProgrammingProblemSolver.getSolution(QuantMatrix, IntervalsArray, size).getResult() == False:
            return InconsistencyResult(False, [])
        else:
            array = LinearProgrammingProblemSolver.getSolution(QuantMatrix, IntervalsArray, size).getArray()
            return LinearProgrammingProblemSolver.getNormalizedSolution(array, size)


class ConjuctInconsistencyChecker(InconsistencyChecker):
    def isInconsistent(self, knowledgePattern):
        size = knowledgePattern.getSize()
        ConjuctMatrix = MatrixProducer.getConjucttoQuantMatrix(size) # что с ним?
        IntervalsArray = knowledgePattern.getArray()
        return LinearProgrammingProblemSolver.getSolution(ConjuctMatrix, IntervalsArray, size)


class DisjunctInconsistencyChecker(InconsistencyChecker):
    def isInconsistent(self, knowledgePattern):         # тут должна быть дизъюнктная штука слышешьб
        size = knowledgePattern.getSize()
        DisjunctMatrix = MatrixProducer.getDisjuncttoQuantMatrix(size)  #что с ним?
        IntervalsArray = knowledgePattern.getArray()
        return LinearProgrammingProblemSolver.getSolution(DisjunctMatrix, IntervalsArray, size)

class MatrixProducer:
    def getDisjuncttoQuantMatrix(self, size): ### возвращает элемент класса диаграмма
        return np.linalg.inv(self.getQuanttoDisjunctMatrix(math.log(size, 2)))

    def getQuanttoDisjunctMatrix(self, n):            #self?
        if n == 0:
            return np.array([1], dtype=np.double)
        elif n == 1:
            return np.array([[1, 1], [0, 1]], dtype=np.double)
        else:
            K = self.getQuanttoDisjunctMatrix(n-1)          #???
            I = np.ones((2 ** (n - 1), 2 ** (n - 1)), dtype=np.double)
            K_o = K.copy()
            K_o[0] = [0] * 2 ** (n - 1)
            return np.block([[K, K], [K_o, I]])

    def getConjucttoQuantMatrix(self, n):
        if n == 0:
            return np.array([1], dtype=np.double)
        elif n == 1:
            return np.array([[1, -1], [0, 1]], dtype=np.double)
        else:
            I = self.getConjucttoQuantMatrix(n-1)          #???
            O = np.zeros((2 ** (n - 1), 2 ** (n - 1)), dtype=np.double)
            return np.block([[I, (-1)*I], [O, I]])

    def getQuantMatrix(self, size):
        return np.eye(size, dtype=np.double)




class LinearProgrammingProblemSolver:
    def getSolution(self, matrixs, array, size):
        A = np.vstack(((-1) * matrixs, (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        A = matrix(A)
        B = np.hstack((np.zeros(size, dtype=np.double), (-1) * array[:, 0], array[:, 1]))
        B = matrix(B)
        c = np.array(np.zeros(size, dtype=np.double))
        c = matrix(c)
        solvers.options['show_progress'] = False
        flagNone = 0
        resultArray = array.copy()
        for i in range(size):
            c[i] = 1
            sol = solvers.lp(c, A, B)
            if sol['x'] is None:
                flagNone = 1
                resultArray = []                                      
                break
            resultArray[i][0] = round(sol['x'][i], 3)                              # тут надо делать копию или оставлять прежним?
            c[i] = -1
            sol = solvers.lp(c, A, B)
            if sol['x'] is None:
                flagNone = 1
                break
            resultArray[i][1] = round(sol['x'][i], 3)
            c[i] = 0
        return InconsistencyResult(not(flagNone), resultArray)

    def getNormalizedSolution(self, array, size):
        A = np.vstack(((-1) * np.ones(size, dtype=np.double), np.ones(size, dtype=np.double), (-1) * np.eye(size, dtype=np.double), np.eye(size, dtype=np.double)))
        A = matrix(A)
        B = np.hstack(((-1) * np.ones(1, dtype=np.double), np.ones(1, dtype=np.double), (-1) * array[:, 0], array[:, 1]))
        B = matrix(B)
        c = np.array(np.zeros(size, dtype=np.double))
        c = matrix(c)
        flagNone = 0
        resultArray = array.copy()
        for i in range(size):
            c[i] = 1
            sol = solvers.lp(c, A, B)
            if sol['x'] is None:
                flagNone = 1
                resultArray = []
                break
            resultArray[i][0] = round(sol['x'][i], 3)  # тут надо делать копию?
            c[i] = -1
            sol = solvers.lp(c, A, B)
            if sol['x'] is None:
                flagNone = 1
                break
            resultArray[i][1] = round(sol['x'][i], 3)
            c[i] = 0
        return InconsistencyResult(not(flagNone), resultArray)




class InconsistencyResult:

    def __init__(self, verdict, arr):
        self.verdict = verdict
        self.arr = arr

    def getArray(self):
        if verdict == True:
            return self.arr
        else:
            raise AttributeError


    def getResult(self):
        return self.verdict



class KnowledgePatternItem:
    def __init__(self, arr, type):
        self.type = type
        self.arr = arr

    def getType(self):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")

    def getElement(self, index):
        raise NotImplementedError("It's a method of abstract class, use appropriate implementation")

    def getArray(self):
        return NotImplementedError("It's a method of abstract class, use appropriate implementation")

    def getSize(self):
        return NotImplementedError("It's a method of abstract class, use appropriate implementation")

class QuantKnowledgePatternItem(KnowledgePatternItem):
    def getType(self):
        return self.type

    def getElement(self, index):
        return self.arr[index]

    def getArray(self):
        return self.arr

    def getSize(self):
        return len(self.arr)


class DisjunctKnowledgePatternItem(KnowledgePatternItem):
    def getType(self):
        return self.type

    def getElement(self, index):
        return self.arr[index]

    def getArray(self):
        return self.arr

    def getSize(self):
        return len(self.arr)

class ConjuctKnowledgePatternItem(KnowledgePatternItem):
    def getType(self):
        return self.type

    def getElement(self, index):
        return self.arr[index]

    def getArray(self):
        return self.arr

    def getSize(self):
        return len(self.arr)

