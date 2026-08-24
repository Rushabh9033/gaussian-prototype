from .psf_estimator import estimate_psf
from .forward_model import forward_model
from .backprojection import run_scpb_optimization
from .regularization import compute_regularization_energy
from .tiling import reconstruct_tiled
from .metrics import (
    color_histogram_distance,
    compute_tiling_equivalence_error,
    edge_f1_score,
    calculate_ringing_percentage,
    compute_laplacian_variance,
    compute_gradient_energy
)

configs = {
    "SCPB_A": {
        "psf_multiplier": 0.75,
        "max_iters": 6,
        "initial_step": 0.40,
        "reg_weight": 0.05,
        "overshoot_weight": 20.0
    },
    "SCPB_B": {
        "psf_multiplier": 1.00,
        "max_iters": 10,
        "initial_step": 0.55,
        "reg_weight": 0.01,
        "overshoot_weight": 10.0
    },
    "SCPB_C": {
        "psf_multiplier": 1.25,
        "max_iters": 15,
        "initial_step": 0.65,
        "reg_weight": 0.001,
        "overshoot_weight": 5.0
    }
}
