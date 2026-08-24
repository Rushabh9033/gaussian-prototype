import numpy as np

def extract_source_residuals(source_linear: np.ndarray, base_1x_linear: np.ndarray) -> np.ndarray:
    """
    Computes the source-supported residual information.
    R = Source - Base
    """
    return source_linear - base_1x_linear

def evaluate_residual(residual: np.ndarray, target_shape: tuple, method) -> np.ndarray:
    """
    Evaluates the residual pyramid at the requested scale (upsampling).
    We do not invent high-frequency bands, we just interpolate the source-supported residual.
    """
    # method is expected to be a callable like cv2.resize
    return method(residual, target_shape)
