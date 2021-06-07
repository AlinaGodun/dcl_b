import torch


def dec_prediction(centers, embedded, alpha=1.0):
    ta = centers.unsqueeze(0)
    tb = embedded.unsqueeze(1)
    squared_diffs = (ta - tb).pow(2).sum(2)
    numerator = (1.0 + squared_diffs / alpha).pow(-1.0 * (alpha + 1.0) / 2.0)
    denominator = numerator.sum(1)
    prob = numerator / denominator.unsqueeze(1)
    return prob


def dec_compression_value(pred_labels):
    soft_freq = pred_labels.sum(0)
    squared_pred = pred_labels.pow(2)
    normalized_squares = squared_pred / soft_freq.unsqueeze(0)
    sum_normalized_squares = normalized_squares.sum(1)
    p = normalized_squares / sum_normalized_squares.unsqueeze(1)
    return p


def dec_compression_loss_fn(q_clean, q_augs=[]):
    ## 1 q_clean = dec_prediction with clean data
    ## 2 p_clean = dec compression data with clean data
    ## 3 q_aug = dec prediction with aug data
    ## 4 loss_q_clean_p_clean
    ## 5 loss q_aug_p_clean
    ## 6 loss = (loss_q_clean_p_clean + q_aug_p_clean) / n

    p_clean = dec_compression_value(q_clean).detach()
    ## take average of p_clean
    loss = get_loss(p_clean, q_clean)

    for q_aug in q_augs:
        loss += get_loss(p_clean, q_aug)

    return loss / (len(q_augs) + 1)


def get_loss(p, q):
    return -1.0 * torch.mean(torch.sum(p * torch.log(q + 1e-8), dim=1))
