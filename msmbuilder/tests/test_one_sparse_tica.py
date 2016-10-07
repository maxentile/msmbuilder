import numpy as np
from msmbuilder.decomposition import OneSparseTICA
from msmbuilder.example_datasets import AlanineDipeptide

def build_dataset():
    ''' overcomplete representation of alanine dipeptide'''
    trajs = AlanineDipeptide().get().trajectories

    pairs = []
    for i in range(trajs[0].n_atoms):
        for j in range(i):
            pairs.append((i, j))

    from msmbuilder.featurizer import AtomPairsFeaturizer
    apf = AtomPairsFeaturizer(pairs)
    X = apf.fit_transform(trajs)
    return X

def test_alanine():
    '''
    - make sure we can recover at least 15 components from this dataset with no linalg errors
    - make sure dominant component corresponds to atom pair 142
    '''

    X = build_dataset()

    o_sptica = OneSparseTICA(n_components=15, lag_time=10)
    _ = o_sptica.fit_transform(X)

    np.testing.assert_(o_sptica.components_[0][142]==1)