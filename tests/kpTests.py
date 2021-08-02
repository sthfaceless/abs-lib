import unittest

import numpy as np

from abslib.kp import KnowledgePatternManager, DisjunctKnowledgePatternItem, MatrixProducer, QuantKnowledgePatternItem, \
    ConjunctKnowledgePatternItem


# Tests for knowledge pattern part of abslib
class KnowledgePatternManagerTest(unittest.TestCase):

    def testDisjunctsConsistent(self):
        arrays = [[[1.0, 1.0], [0.1, 0.2], [0.2, 0.4], [0.5, 0.7]]]
        for disjunct_intervals_consistent in arrays:
            knowledgePattern = DisjunctKnowledgePatternItem(disjunct_intervals_consistent)
            result = KnowledgePatternManager.checkConsistency(knowledgePattern)
            self.assertTrue(result.consistent, "False negative consistency result")
            self.assertTrue(np.array(result.array).shape == np.array(disjunct_intervals_consistent).shape,
                            "Incorrect result array size")
            for i in range(len(result.array)):
                self.assertTrue(disjunct_intervals_consistent[i][0] <= result.array[i][0]
                                and result.array[i][1] <= disjunct_intervals_consistent[i][1],
                                "Intervals couldn't become larger")

    def testDisjunctsInconsistent(self):
        arrays = [[[1, 1], [0.1, 0.2], [0.2, 0.4], [0.7, 0.7]]]
        for disjunct_intervals_inconsistent in arrays:
            knowledgePattern = DisjunctKnowledgePatternItem(disjunct_intervals_inconsistent)
            result = KnowledgePatternManager.checkConsistency(knowledgePattern)
            self.assertFalse(result.consistent, "False positive consistency result")

    def testQuantsConsistent(self):
        arrays = [[[0.24, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]]
        for quant_intervals_consistent in arrays:
            knowledgePattern = QuantKnowledgePatternItem(quant_intervals_consistent)
            result = KnowledgePatternManager.checkConsistency(knowledgePattern)
            self.assertTrue(result.consistent, "False negative consistency result")
            self.assertTrue(np.array(result.array).shape == np.array(quant_intervals_consistent).shape,
                            "Incorrect result array size")
            for i in range(len(result.array)):
                self.assertTrue(quant_intervals_consistent[i][0] <= result.array[i][0]
                                and result.array[i][1] <= quant_intervals_consistent[i][1],
                                "Intervals couldn't become larger")

    def testQuantsInconsistent(self):
        arrays = [[[0.2, 0.3], [0.2, 0.3], [0.2, 0.3], [0.6, 0.7]]]
        for quant_intervals_inconsistent in arrays:
            knowledgePattern = QuantKnowledgePatternItem(quant_intervals_inconsistent)
            result = KnowledgePatternManager.checkConsistency(knowledgePattern)
            self.assertFalse(result.consistent, "False positive consistency result")

    def testConjunctsConsistent(self):
        arrays = [[[1.0, 1.0], [0.6, 0.9], [0.6, 0.9], [0.2, 0.3]]]
        for conjunct_intervals_consistent in arrays:
            knowledgePattern = ConjunctKnowledgePatternItem(conjunct_intervals_consistent)
            result = KnowledgePatternManager.checkConsistency(knowledgePattern)
            self.assertTrue(result.consistent, "False negative consistency result")
            self.assertTrue(np.array(result.array).shape == np.array(conjunct_intervals_consistent).shape,
                            "Incorrect result array size")
            for i in range(len(result.array)):
                self.assertTrue(conjunct_intervals_consistent[i][0] <= result.array[i][0]
                                and result.array[i][1] <= conjunct_intervals_consistent[i][1],
                                "Intervals couldn't become larger")

    def testConjunctsInconsistent(self):
        arrays = [[[1, 1], [0.1, 0.2], [0.2, 0.4], [0.8, 0.8]]]
        for conjunct_intervals_consistent in arrays:
            knowledgePattern = DisjunctKnowledgePatternItem(conjunct_intervals_inconsistent)
            result = KnowledgePatternManager.checkConsistency(knowledgePattern)
            self.assertFalse(result.consistent, "False positive consistency result")

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
