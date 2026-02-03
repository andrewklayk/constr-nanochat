import torch

def fand(alpha, beta=None):
    if beta:
        return torch.min(alpha, beta)
    return torch.min(alpha, dim=-1)

def fneg(alpha):
    return 1 - alpha

def fimpl(alpha, beta):
    # if alpha == 0:
    #     return torch.ones_like(beta)
    return torch.min(torch.ones_like(beta), beta/alpha)

def pos_impl_constraint(probs1, prob2):
    """
    Compute a regularization loss term that penalizes fuzzy implication of a specific token.

    Args:
        probs1: Probabilities of the left-side of the implication. Will be conjuncted.
        prob2: Probability of the right-side.

    Returns:
        A scalar loss term to be added to the total loss
    """
    if probs1.ndim > 1 and probs1.shape[-1] > 1:
        lh = fand(probs1)
    else:
        lh = probs1
    
    rh = prob2
    
    imp = fimpl(lh, rh)
    return imp

def implication_constraint(inputs, lhs_symbols_idx, rhs_symbol_idx):
    probs = torch.nn.functional.softmax(inputs, dim=-1)
    for pos in range(probs.shape[-2]-1):
        con += torch.mean(pos_impl_constraint(probs[:,pos,lhs_symbols_idx], probs[:, pos+1, rhs_symbol_idx])  - 1, dim=0)