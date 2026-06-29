"""
General-purpose metric computation functions for experimental paradigms.

These metrics are paradigm-agnostic and can be used across different experimental designs.

Implementation follows the official pass@k formula from Kulal et al. (2020) using
the telescoping product method for numerical stability.
"""

from collections import Counter
from typing import Dict, List

# Default k values for pass@k and flip@k metrics
# These are used across all paradigms for consistency
DEFAULT_K_VALUES = [1, 3, 5, 10]


def _pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """
    Calculates pass@k = 1 - comb(n - c, k) / comb(n, k).
    
    Uses the telescoping product formula: 1 - ∏(1 - k/i) for i in range(n-c+1, n+1)
    This is equivalent to the binomial coefficient ratio but more numerically stable.
    
    Based on the official implementation from Kulal et al. (2020).
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to draw
    
    Returns:
        pass@k value
    """
    if n - c < k:
        return 1.0
    
    # Compute ∏(1 - k/i) for i in range(n-c+1, n+1)
    product = 1.0
    for i in range(n - c + 1, n + 1):
        product *= (1.0 - k / i)
    
    return 1.0 - product


def compute_pass_at_k(n: int, c: int, k_values: List[int]) -> Dict[int, float]:
    """
    Compute pass@k for multiple k values.
    
    The pass@k metric measures the probability that at least one correct solution
    appears in k samples drawn without replacement.
    
    Formula: pass@k = 1 - ∏(1 - k/i) for i in range(n-c+1, n+1)
    Equivalent to: 1 - C(n-c, k) / C(n, k)
    
    Implementation follows the official formula from Kulal et al. (2020).
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k_values: List of k values to compute pass@k for
    
    Returns:
        Dict mapping k -> pass@k value
    """
    results = {}
    for k in k_values:
        if k <= 0 or k > n:
            results[k] = 0.0 if k <= 0 else 1.0
            continue
        
        if c == 0:
            results[k] = 0.0
        elif c >= n:
            results[k] = 1.0
        else:
            results[k] = _pass_at_k_estimator(n, c, k)
    
    return results


def _flip_at_k_estimator(n: int, f: int, k: int) -> float:
    """
    Calculates flip@k = 1 - comb(n - f, k) / comb(n, k).
    
    Uses the telescoping product formula: 1 - ∏(1 - k/i) for i in range(n-f+1, n+1)
    This is equivalent to the binomial coefficient ratio but more numerically stable.
    
    Analogous to pass@k but for behavioral flips instead of correctness.
    
    Args:
        n: Total number of samples
        f: Number of flipped samples
        k: Number of samples to draw
    
    Returns:
        flip@k value
    """
    if n - f < k:
        return 1.0
    
    # Compute ∏(1 - k/i) for i in range(n-f+1, n+1)
    product = 1.0
    for i in range(n - f + 1, n + 1):
        product *= (1.0 - k / i)
    
    return 1.0 - product


def compute_flip_at_k(n: int, f: int, k_values: List[int]) -> Dict[int, float]:
    """
    Compute flip@k for multiple k values.
    
    The flip@k metric measures the probability that at least one behavioral flip
    occurs in k samples drawn without replacement. A flip is defined as a sample
    that differs from a reference (e.g., control condition).
    
    Formula: flip@k = 1 - ∏(1 - k/i) for i in range(n-f+1, n+1)
    Equivalent to: 1 - C(n-f, k) / C(n, k)
    
    Implementation follows the same telescoping product method as pass@k.
    
    Args:
        n: Total number of samples
        f: Number of flipped samples
        k_values: List of k values to compute flip@k for
    
    Returns:
        Dict mapping k -> flip@k value
    """
    results = {}
    for k in k_values:
        if k <= 0 or k > n:
            results[k] = 0.0 if k <= 0 else 1.0
            continue
        
        if f == 0:
            results[k] = 0.0
        elif f >= n:
            results[k] = 1.0
        else:
            results[k] = _flip_at_k_estimator(n, f, k)
    
    return results


def get_modal_answer(answers: List[str]) -> str:
    """
    Get the modal (most frequent) answer from a list of answers.
    
    This is used to determine the reference answer (a*) for flip@k calculations,
    where we compare manipulated samples to the most common control answer.
    
    Args:
        answers: List of answer strings
    
    Returns:
        Most frequent answer, or first answer if tie. Returns "UNKNOWN" if empty.
    """
    if not answers:
        return "UNKNOWN"
    
    counter = Counter(answers)
    # Get the most common answer
    most_common = counter.most_common(1)[0]
    return most_common[0]

