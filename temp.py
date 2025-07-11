import numpy as np

def normalize_compactness(compactness, c_min=4*np.pi, c_max=200):
    c = np.maximum(compactness, c_min)
    score = 1 - (np.log(c / c_min) / np.log(c_max / c_min))
    return np.clip(score, 0, 1)

def normalize_centroid_dispersion(dispersion, d_min=0, d_max=100):
    # 小离散更好 ⇒ 高得分
    score = 1 - (dispersion - d_min) / (d_max - d_min)
    return np.clip(score, 0, 1)

def compute_shape_score(compactness, solidity, centroid_dispersion,
                        weights=(1.0, 1.0, 1.0),
                        c_max=200,
                        dispersion_range=(0, 100)):
    """
    Compute a total shape score based on 3 indicators:
    - Compactness (lower is better)
    - Solidity (higher is better)
    - Centroid Dispersion (lower is better)
    """

    # Normalize individual scores
    c_min,c_max = 4*np.pi,
    c = np.maximum(compactness, c_min)
    score = 1 - (np.log(c / c_min) / np.log(c_max / c_min))
    compactness_score = normalize_compactness(compactness, *compactness_range)
    solidity_score = np.clip(solidity, 0, 1)
    dispersion_score = normalize_centroid_dispersion(centroid_dispersion, *dispersion_range)

    # Apply weights
    w1, w2, w3 = weights
    total_weight = w1 + w2 + w3
    final_score = (w1 * compactness_score + w2 * solidity_score + w3 * dispersion_score) / total_weight

    return {
        'compactness_score': compactness_score,
        'solidity_score': solidity_score,
        'dispersion_score': dispersion_score,
        'final_score': final_score
    }