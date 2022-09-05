# Hungarian method
from munkres import Munkres
hungarian = Munkres()


def hungarian_match(cost_matrix):
    assert cost_matrix.shape[0] == cost_matrix.shape[1]
    M = cost_matrix.copy()
    indices = hungarian.compute(M.tolist())

    j2i = {j:i for i,j in indices}
    paired_is = [j2i[j] for j in range(len(j2i))]

    return paired_is