"""Matrix operations for connectivity analysis."""

import numpy as np
from typing import Dict, List, Tuple

from nilearn.connectome import ConnectivityMeasure


# All supported connectivity kinds
CONNECTIVITY_KINDS = ['correlation', 'covariance', 'partial correlation', 'precision']


def sym_matrix_to_vec(matrix: np.ndarray) -> np.ndarray:
    """Convert symmetric matrix to vector (upper triangle).
    
    Extracts the upper triangle of a symmetric matrix (excluding diagonal)
    and returns it as a 1D vector.
    
    Args:
        matrix: Symmetric matrix of shape (N, N)
    
    Returns:
        1D vector of length N*(N-1)/2 containing upper triangle values
    
    Raises:
        ValueError: If matrix is not square
    
    Example:
        >>> matrix = np.array([[0, 1, 2],
        ...                    [1, 0, 3],
        ...                    [2, 3, 0]])
        >>> vec = sym_matrix_to_vec(matrix)
        >>> vec
        array([1, 2, 3])
    """
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape {matrix.shape}")
    
    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
    
    # Extract upper triangle indices (excluding diagonal)
    indices = np.triu_indices(n_rows, k=1)
    
    # Extract values
    vector = matrix[indices]
    
    return vector


def vec_to_sym_matrix(vector: np.ndarray, n: int) -> np.ndarray:
    """Convert vector to symmetric matrix.
    
    Reconstructs a symmetric matrix from its upper triangle vector.
    
    Args:
        vector: 1D vector of length N*(N-1)/2
        n: Size of the output matrix (N x N)
    
    Returns:
        Symmetric matrix of shape (N, N) with zeros on diagonal
    
    Raises:
        ValueError: If vector length doesn't match expected size
    
    Example:
        >>> vec = np.array([1, 2, 3])
        >>> matrix = vec_to_sym_matrix(vec, 3)
        >>> matrix
        array([[0, 1, 2],
               [1, 0, 3],
               [2, 3, 0]])
    """
    expected_length = n * (n - 1) // 2
    if len(vector) != expected_length:
        raise ValueError(
            f"Vector length {len(vector)} doesn't match expected size "
            f"{expected_length} for {n}x{n} matrix"
        )
    
    # Create empty matrix
    matrix = np.zeros((n, n))
    
    # Get upper triangle indices
    indices = np.triu_indices(n, k=1)
    
    # Fill upper triangle
    matrix[indices] = vector
    
    # Make symmetric
    matrix = matrix + matrix.T
    
    return matrix


def compute_connectivity_matrix(
    time_series: np.ndarray,
    kind: str = "correlation"
) -> np.ndarray:
    """Compute connectivity matrix from time series.
    
    Args:
        time_series: Time series array of shape (n_timepoints, n_regions)
        kind: Type of connectivity measure:
            - 'correlation': Pearson correlation (normalized covariance)
            - 'covariance': Sample covariance (NOT standardized)
            - 'partial correlation': Correlation controlling for other regions
            - 'precision': Inverse covariance (sparse direct connections)
    
    Returns:
        Connectivity matrix of shape (n_regions, n_regions)
    
    Raises:
        ValueError: If time_series has wrong shape or kind is invalid
    
    Example:
        >>> ts = np.random.randn(100, 5)  # 100 timepoints, 5 regions
        >>> conn = compute_connectivity_matrix(ts)
        >>> conn.shape
        (5, 5)
        >>> np.allclose(conn, conn.T)  # Symmetric
        True
    """
    if time_series.ndim != 2:
        raise ValueError(
            f"time_series must be 2D (timepoints x regions), "
            f"got shape {time_series.shape}"
        )
    
    if kind not in CONNECTIVITY_KINDS:
        raise ValueError(
            f"Unknown connectivity kind: '{kind}'. "
            f"Supported: {CONNECTIVITY_KINDS}"
        )
    
    # Use nilearn's ConnectivityMeasure for robust computation
    # For covariance, do NOT standardize (otherwise cov == corr)
    # For correlation-based measures, standardize for numerical stability
    if kind == 'covariance':
        standardize = False
    else:
        standardize = 'zscore_sample'
    
    conn_measure = ConnectivityMeasure(kind=kind, standardize=standardize)
    
    # ConnectivityMeasure expects list of subjects, each with shape (n_samples, n_features)
    connectivity = conn_measure.fit_transform([time_series])[0]
    
    # Set diagonal to zero for correlation-like measures
    if kind in ['correlation', 'partial correlation']:
        np.fill_diagonal(connectivity, 0)
    
    return connectivity


def compute_all_connectivity_matrices(
    time_series: np.ndarray,
    kinds: List[str] = None
) -> Dict[str, np.ndarray]:
    """Compute all connectivity matrices from time series.
    
    Args:
        time_series: Time series array of shape (n_timepoints, n_regions)
        kinds: List of connectivity kinds to compute. If None, computes all.
    
    Returns:
        Dictionary mapping kind name to connectivity matrix
    
    Example:
        >>> ts = np.random.randn(100, 5)
        >>> matrices = compute_all_connectivity_matrices(ts)
        >>> list(matrices.keys())
        ['correlation', 'covariance', 'partial correlation', 'precision']
    """
    if kinds is None:
        kinds = CONNECTIVITY_KINDS
    
    matrices = {}
    for kind in kinds:
        matrices[kind] = compute_connectivity_matrix(time_series, kind=kind)
    
    return matrices


def fisher_z_transform(correlation_matrix: np.ndarray) -> np.ndarray:
    """Apply Fisher z-transformation to correlation matrix.
    
    Transforms correlation coefficients to z-scores using Fisher's
    transformation: z = 0.5 * ln((1+r)/(1-r))
    
    Args:
        correlation_matrix: Matrix of correlation coefficients
    
    Returns:
        Fisher z-transformed matrix
    
    Note:
        Correlations of exactly +1 or -1 will be clipped to avoid infinities
    """
    # Clip to avoid division by zero and infinities
    r_clipped = np.clip(correlation_matrix, -0.999999, 0.999999)
    
    # Apply Fisher z-transformation
    z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    
    return z


def inverse_fisher_z_transform(z_matrix: np.ndarray) -> np.ndarray:
    """Apply inverse Fisher z-transformation.
    
    Transforms z-scores back to correlation coefficients:
    r = (exp(2*z) - 1) / (exp(2*z) + 1)
    
    Args:
        z_matrix: Matrix of Fisher z-scores
    
    Returns:
        Matrix of correlation coefficients
    """
    # Apply inverse transformation
    exp_2z = np.exp(2 * z_matrix)
    r = (exp_2z - 1) / (exp_2z + 1)
    
    return r
