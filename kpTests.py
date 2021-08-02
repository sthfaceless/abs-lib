import unittest
import numpy as np
from abslib.kp import KnowledgePatternManager, DisjunctKnowledgePatternItem, MatrixProducer, QuantKnowledgePatternItem, ConjunctKnowledgePatternItem


# Tests for verification that the knowledge pattern is a part of abslib
class KnowledgePatternManagerTest(unittest.TestCase):

    def testDisjunctsInconsistent(self):
        arrays = [[[1, 1], [0.1, 0.2], [0.2, 0.4], [0.5, 0.7]],
                  [[1, 1], [0.3, 0.4], [0.53, 0.7], [0.8, 0.82]],  # 0.3, 0.7,  0.8
                  [[1, 1], [0.45, 0.54], [0.32, 0.41], [0.6, 0.6], [0.57, 0.7], [0.55, 0.62], [0.68, 0.7], [0.9, 0.99]],  # 0.5, 0.4, 0.6, 0.6, 0.6, 0.7, 0.9
                  [[1, 1], [], [], [], [], [], [], []],
                  [[1, 1], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
                  [[1, 1], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]
        for disjunct_intervals_inconsistent in arrays:
            knowledgePattern = DisjunctKnowledgePatternItem(disjunct_intervals_inconsistent)
            result = KnowledgePatternManager.checkInconsistency(knowledgePattern)
            self.assertTrue(result.inconsistent, "False negative inconsistency result")
            self.assertTrue(np.array(result.array).shape == np.array(disjunct_intervals_inconsistent).shape,
                            "Incorrect result array size")
            for i in range(len(result.array)):
                self.assertTrue(disjunct_intervals_inconsistent[i][0] <= result.array[i][0]
                                and result.array[i][1] <= disjunct_intervals_inconsistent[i][1],
                                "Intervals couldn't become larger")
    def testDisjunctsNotInconsistent(self):
        arrays = [[[1, 1], [0.1, 0.2], [0.2, 0.4], [0.7, 0.7]],
                  [[1, 1], [0.5, 0.6], [0.3, 0.9], [0.3, 0.4]],
                  [[1, 1], [], [], [], [], [], [], []],
                  [[1, 1], [], [], [], [], [], [], []],
                  [[1, 1], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
                  [[1, 1], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]
        for disjunct_intervals_inconsistent in arrays:
            knowledgePattern = DisjunctKnowledgePatternItem(disjunct_intervals_inconsistent)
            result = KnowledgePatternManager.checkInconsistency(knowledgePattern)
            self.assertFalse(result.inconsistent, "False positive inconsistency result")


    def testConjunctsInconsistent(self):
        arrays = [[[1, 1], [0.6, 0.9], [0.6, 0.9], [0.2, 0.3]],
                  [[1, 1], [0.56, 0.78], [0.23, 0.9], [0.3, 1.0]],   # 0.7, 0.8,  0.7
                  [[1, 1], [0.5, 0.6], [0.3, 0.9], [0.3, 0.4]],   # 0.6, 0.9,  0.3
                  [[1, 1], [0.78, 0.95], [0.2, 0.6], [0.3, 0.44], [0.0, 0.2], [0.1, 0.4], [0.15, 0.7], [0.03, 0.28]],  # 0.8, 0.3, 0.4,  0.2, 0.2, 0.2,  0.1
                  [[1, 1], [0.7, 0.87], [0.32, 0.9], [0.3, 0.55], [0.32, 0.49], [0.2, 0.7], [0.0, 0.25], [0.13, 0.56]],  # 0.8, 0.4, 0.5,  0.4, 0.3, 0.2,  0.2
                  [[1, 1], [0.9, 0.98], [0.18, 0.85], [0.4, 0.41], [0.24, 0.44], [0.35, 0.5], [0.35, 0.75], [0.23, 0.37], [0.14, 0.56], [0.35, 0.36], [0.3, 0.39], [0.03, 0.3], [0.0, 0.42], [0.15, 0.4], [0.08, 0.22], [0.0, 0.11]],
                  [[1, 1], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
                  [[1, 1], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]
        for conjunct_intervals_inconsistent in arrays:
            knowledgePattern = ConjunctKnowledgePatternItem(conjunct_intervals_inconsistent)
            result = KnowledgePatternManager.checkInconsistency(knowledgePattern)
            self.assertTrue(result.inconsistent, "False negative inconsistency result")
            self.assertTrue(np.array(result.array).shape == np.array(conjunct_intervals_inconsistent).shape,
                            "Incorrect result array size")
            for i in range(len(result.array)):
                self.assertTrue(conjunct_intervals_inconsistent[i][0] <= result.array[i][0]
                                and result.array[i][1] <= conjunct_intervals_inconsistent[i][1],
                                "Intervals couldn't become larger")
    def testConjunctsNotInconsistent(self):
        arrays = [[[1, 1], [0.1, 0.2], [0.2, 0.4], [0.8, 0.8]],
                  [[1, 1], [], [], []],
                  [[1, 1], [], [], []],
                  [[1, 1], [], [], [], [], [], [], []],
                  [[1, 1], [], [], [], [], [], [], []],
                  [[1, 1], [0.9, 0.98], [0.18, 0.85], [0.4, 0.41], [0.24, 0.44], [0.35, 0.5], [0.35, 0.75], [0.3, 0.3], [0.4, 0.4], [0.35, 0.35], [0.32, 0.32], [0.3, 0.3], [0.31, 0.31], [0.25, 0.25], [0.2, 0.2], [0.0, 0.0]],
                  [[1, 1], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]]
        for conjunct_intervals_inconsistent in arrays:
            knowledgePattern = ConjunctKnowledgePatternItem(conjunct_intervals_inconsistent)
            result = KnowledgePatternManager.checkInconsistency(knowledgePattern)
            self.assertFalse(result.inconsistent, "False positive inconsistency result")


    def testQuantsInconsistent(self):
        arrays = [[[0.24, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]],
                   [[0.3, 1.0], [0.5, 0.7], [0.1, 0.5], [0.1, 0.8]]]
        for quant_intervals_inconsistent in arrays:
            knowledgePattern = QuantKnowledgePatternItem(quant_intervals_inconsistent)
            result = KnowledgePatternManager.checkInconsistency(knowledgePattern)
            self.assertTrue(result.inconsistent, "False negative inconsistency result")
            self.assertTrue(np.array(result.array).shape == np.array(quant_intervals_inconsistent).shape,
                            "Incorrect result array size")
            for i in range(len(result.array)):
                self.assertTrue(quant_intervals_inconsistent[i][0] <= result.array[i][0]
                                and result.array[i][1] <= quant_intervals_inconsistent[i][1],
                                "Intervals couldn't become larger")
    def testQuantsNotInconsistent(self):
        arrays = [[[0.2, 0.3], [0.2, 0.3], [0.2, 0.3], [0.6, 0.7]]]
        for quant_intervals_inconsistent in arrays:
            knowledgePattern = QuantKnowledgePatternItem(quant_intervals_inconsistent)
            result = KnowledgePatternManager.checkInconsistency(knowledgePattern)
            self.assertFalse(result.inconsistent, "False positive inconsistency result")


    def testDisjunctsToQuantsMatrix(self):
        matrices = [(np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0],
                               [-0.0, -0.0, -0.0, -0.0, -1.0, 1.0, 1.0, -1.0],
                               [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                               [-0.0, -0.0, -1.0, 1.0, -0.0, -0.0, 1.0, -1.0],
                               [-0.0, -1.0, -0.0, 1.0, -0.0, 1.0, -0.0, -1.0],
                               [0.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]], dtype=np.double), 3)]
        for matrix, n in matrices:
            generated_matrix = MatrixProducer.getDisjunctsToQuantsMatrix(n)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    self.assertEqual(matrix[i][j], generated_matrix[i][j], "Wrong matrix generation algorithm")

    def testConjunctsToQuantsMatrix(self):
        matrices = [(np.array([[1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
                               [0.0, 1.0, -0.0, -1.0, -0.0, -1.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0, -1.0, -0.0, -0.0, -1.0, 1.0],
                               [0.0, 0.0, 0.0, 1.0, -0.0, -0.0, -0.0, -1.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.0, -1.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.double), 3)]
        for matrix, n in matrices:
            generated_matrix = MatrixProducer.getConjunctsToQuantsMatrix(n)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    self.assertEqual(matrix[i][j], generated_matrix[i][j], "Wrong matrix generation algorithm")

    def testQuantsToDisjunctsMatrix(self):
        matrices = [(np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                               [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                               [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.double), 3)]
        for matrix, n in matrices:
            generated_matrix = MatrixProducer.getQuantsToDisjunctsMatrix(n)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    self.assertEqual(matrix[i][j], generated_matrix[i][j], "Wrong matrix generation algorithm")


if __name__ == '__main__':
    unittest.main()