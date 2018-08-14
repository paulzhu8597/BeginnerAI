import torch

def decode(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    return point_form(boxes)

def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)

def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = torch.zeros(scores.size(0)).long()
    if boxes.numel() == 0:
        return keep, 0
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    count = 0
    while idx.numel() > 0:
        i = idx[0]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[1:]
        xx1 = x1.index_select(0, idx).clamp_(min=x1[i].item())
        yy1 = y1.index_select(0, idx).clamp_(min=y1[i].item())
        xx2 = x2.index_select(0, idx).clamp_(max=x2[i].item())
        yy2 = y2.index_select(0, idx).clamp_(max=y2[i].item())
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        rem_areas = area.index_select(0, idx)
        union = rem_areas - inter + area[i]
        iou = inter / union
        idx = idx[iou.le(overlap)]
    return keep, count

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(truths, point_form(priors))
    # [num_obj] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    # [num_prior] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    # ensure ground truth's "best priors" won't be "delete"
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # [num_priors, 4]
    conf = labels[best_truth_idx] + 1  # [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as bkg
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf

# calculate (A∩B)/(A∪B) --- return size [mxn]
def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


# Encode the variances from the priorbox layers into the ground truth boxes
# matched---point form, priors---center form
def encode(matched, priors, variances):
    g_xy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_xy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat((g_xy, g_wh), 1)  # [num_priors, 4]

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    iter = torch.clamp(max_xy - min_xy, min=0)
    return iter[:, :, 0] * iter[:, :, 1]

def log_sum_exp(x):
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 2, keepdim=True)) + x_max