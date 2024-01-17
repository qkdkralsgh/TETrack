from torch.nn import functional as F


def ratio_loss(ratio, token_pred_score):
    pred_loss = 0.0

    for i, score in enumerate(token_pred_score):
        pos_ratio = score.mean(1)
        pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()
        
    return pred_loss


def KL_loss(t_feat, s_feat):
    kl_loss = F.kl_div(
        # F.log_softmax(t_feat, dim=-1),
        # F.log_softmax(s_feat, dim=-1),
        t_feat,
        s_feat,
        reduction='batchmean',
        log_target=True
    )
    return kl_loss