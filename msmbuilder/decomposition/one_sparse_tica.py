from __future__ import print_function, division, absolute_import
from six import PY2
import numpy as np
from msmbuilder.decomposition.tica import tICA
from msmbuilder.decomposition._speigh import scdeflate

__all__ = ['OneSparseTICA']

class OneSparseTICA(tICA):
    """
    A special case of approximate time-structure Independent Component Analysis (tICA),
    where each approximate eigenvector is forced to have exactly 1 nonzero value.

    This case has the nice properties that:
    - The resulting tICA components are just as interpretable as the inputs
    - There are no additional free parameters
    - Each 1-sparse approximate eigenvector can be recovered in linear time

    .. warning::

        This model is currently  experimental, and may undergo significant
        changes or bug fixes in upcoming releases.

    .. note::

        Haven't tested different deflation methods here yet.

    Parameters
    ----------
    n_components : int
        Number of sparse tICs to find.
    lag_time : int
        Time-lagged correlations are computed between X[t] and X[t+lag_time].
    shrinkage : float, default=None
        The covariance shrinkage intensity (range 0-1).

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.
    offset_correlation_ : array-like, shape (n_features, n_features)
        Symmetric time-lagged correlation matrix, `C=E[(x_t)^T x_{t+lag}]`.
    eigenvalues_ : array-like, shape (n_features,)
        Psuedo-eigenvalues of the tICA generalized eigenproblem, in decreasing
        order.
    eigenvectors_ : array-like, shape (n_components, n_features)
        Sparse psuedo-eigenvectors of the tICA generalized eigenproblem. The
        vectors give a set of "directions" through configuration space along
        which the system relaxes towards equilibrium.
    means_ : array, shape (n_features,)
        The mean of the data along each feature
    n_observations_ : int
        Total number of data points fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
        online learning.
    n_sequences_ : int
        Total number of sequences fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
         online learning.
    timescales_ : array-like, shape (n_components,)
        The implied timescales of the tICA model, given by
        -offset / log(eigenvalues)

    Algorithm
    ---------
    vs = []
    for i in range(n_components):
        ind = argmax(diag(cov_tau) / diag(cov))
        v = zeros(n_features)
        v[ind] = 1
        vs.append(v)
        cov_tau = deflate(cov_tau)
    return vs

    See Also
    --------
    msmbuilder.decomposition.tICA
    msmbuilder.decomposition.SparseTICA

    References
    ----------
    .. [1] McGibbon, R. T. and V. S. Pande "Identification of simple reaction
        coordinates from complex dynamics." https://arxiv.org/pdf/1602.08776.pdf
    .. [2] Mackey, L. "Deflation Methods for Sparse PCA." NIPS. Vol. 21. 2008.
    .. [3] Noe, F. and Clementi, C. arXiv arXiv:1506.06259 [physics.comp-ph]
           (2015)
    """

    def __init__(self, n_components=None, lag_time=1, kinetic_mapping=True, shrinkage=None):
        super(OneSparseTICA, self).__init__(n_components, lag_time=lag_time,
                                            kinetic_mapping=kinetic_mapping, shrinkage=shrinkage)

    def _one_sparse_eigh_fast(self, A, B):
        '''
        there are only `n_features` candidate vectors in our search space each iteration,
        so here we can pick the best one approximately for free (2 * n_features operations)
        '''
        # compute generalized pseudoeigenvalues of all one-hot vectors
        eigs = np.diag(A) / np.diag(B)

        # return the one-hot vector with the largest (deflated) pseudoeigenvalue
        ind = np.argmax(eigs)
        u = eigs[ind]
        v = np.zeros(len(A))
        v[ind] = 1

        return u, v

    def _solve(self):
        if not self._is_dirty:
            return

        A = self.offset_correlation_
        B = self.covariance_

        eig_vecs = []
        eig_vals = []

        for i in range(self.n_components):
            u, v = self._one_sparse_eigh_fast(A, B)

            eig_vals.append(u)
            eig_vecs.append(v)

            A = scdeflate(A, v)

        # keep it real
        acceptable_inds = np.array([i for i in range(len(eig_vecs)) if not np.isnan(eig_vals[i]) and eig_vals[i] != 0])
        eig_vecs = np.array(eig_vecs)[acceptable_inds]
        eig_vals = np.array(eig_vals)[acceptable_inds]


        # sort in order of decreasing (deflated psuedo)eigenvalue
        self.n_components = min(len(eig_vals), self.n_components)
        self._eigenvalues_ = np.zeros((self.n_components))
        self._eigenvectors_ = np.zeros((self.n_features, self.n_components))

        for i in range(self.n_components):
            self._eigenvalues_[i] = eig_vals[i]
            self._eigenvectors_[:, i] = eig_vecs[i]

        # now we should go back and compute correct eigenvalues -- the accumulated eig_vals here were
        # estimated using the deflated correlation matrices
        # this is a problem also for the current version of SparseTICA
        # In other words:
        # >> sptica = SparseTICA(n_components=5)
        # >> _ = sptica.fit_transform(X)
        # >> assert(sptica.score(X) == sptica.score_)
        # >> False
        num = self._eigenvectors_.T.dot(self.offset_correlation_).dot(self._eigenvectors_)
        denom = self._eigenvectors_.T.dot(self.covariance_).dot(self._eigenvectors_)
        self._eigenvalues_ = np.diag(num.dot(np.linalg.inv(denom)))

        # sort in descending order
        ind = np.argsort(self._eigenvalues_)[::-1]
        self._eigenvalues_ = self._eigenvalues_[ind]
        self._eigenvectors_ = self._eigenvectors_[:, ind]

        self._is_dirty = False

    def summarize(self, n_timescales_to_report=5):
        """Some summary information."""
        nonzeros = np.sum(np.abs(self.eigenvectors_) > 0, axis=0)
        active = '[%s]' % ', '.join(['%d/%d' % (n, self.n_features) for n in nonzeros[:n_timescales_to_report]])

        return """One-sparse time-structure based Independent Components Analysis (tICA)
------------------------------------------------------------------
n_components        : {n_components}
lag_time            : {lag_time}
kinetic_mapping     : {kinetic_mapping}
n_features          : {n_features}
Top {n_timescales_to_report} timescales :
{timescales}
Top {n_timescales_to_report} eigenvalues :
{eigenvalues}
Number of active degrees of freedom:
{active}
""".format(n_components=self.n_components, lag_time=self.lag_time,
           kinetic_mapping=self.kinetic_mapping,
           timescales=self.timescales_[:n_timescales_to_report], eigenvalues=self.eigenvalues_[:n_timescales_to_report],
           n_features=self.n_features, active=active, n_timescales_to_report=n_timescales_to_report)