from abslib.kp import KnowledgePatternManager, ConjunctKnowledgePatternItem, DisjunctKnowledgePatternItem

print(KnowledgePatternManager.checkInconsistency(DisjunctKnowledgePatternItem([[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [0.5, 0.7]])).inconsistent)