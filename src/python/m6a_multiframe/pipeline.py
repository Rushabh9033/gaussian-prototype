from .registration import register_frames
from .robust_fusion import robust_fusion
from .backprojection import iterative_backprojection

def reconstruct_from_frames(frames, scale_factor, configuration):
    """
    Complete multi-frame SR reconstruction pipeline.
    Must not receive ground-truth or true transformations.
    """
    ref_index = configuration.get('ref_index', 0)
    cc_threshold = configuration.get('cc_threshold', 0.85)
    base_sigma = configuration.get('base_sigma', 0.5)
    max_iters = configuration.get('max_iters', 5)
    initial_step = configuration.get('initial_step', 0.1)
    reg_weight = configuration.get('reg_weight', 0.15)
    
    # 1. Registration
    est_translations, accepted = register_frames(frames, ref_index=ref_index, cc_threshold=cc_threshold)
    accepted_ids = [i for i, a in enumerate(accepted) if a]
    
    acc_frames = [frames[i] for i in accepted_ids]
    acc_est = [est_translations[i] for i in accepted_ids]
    
    # 2. Robust fusion
    fused_lin, coverage, confidence = robust_fusion(acc_frames, acc_est, scale_factor)
    
    # 3. Iterative back-projection
    ibp_result, ibp_history = iterative_backprojection(
        fused_lin, acc_frames, acc_est, scale_factor, base_sigma,
        max_iters=max_iters, initial_step=initial_step, reg_weight=reg_weight
    )
    
    return {
        'reconstruction': ibp_result,
        'estimated_translations': est_translations,
        'accepted_ids': accepted_ids,
        'coverage': coverage,
        'confidence': confidence,
        'history': ibp_history
    }
