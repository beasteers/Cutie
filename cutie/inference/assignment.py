import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import logging

log = logging.getLogger(__name__)

class ObjectAssignment:
    def __init__(self, labels=None, allowance_mask=None, max_missed_detection_count=None, new_detection_mask=None, **kw) -> None:
        '''
        
        Arguments:
            track_ids (np.ndarray): shape: [N_input_masks], contains indices that map to track IDs.
            labels ():
            allowance_mask (np.ndarray): shape [N_tracks, N_input_masks], boolean mask for track-input assignment.
        
        '''
        # self.track_ids = track_ids
        self.allowance_mask = allowance_mask
        self.labels = labels
        # self.negative_mask = negative_mask
        self.max_missed_detection_count = max_missed_detection_count
        self.new_detection_mask = new_detection_mask
        self.params = kw

    def __call__(self, pred_prob_with_bg, mask):
        # # select the subset where the masks are not yet assigned
        # track_ids = np.array([-1]*len(mask) if self.track_ids is None else self.track_ids)
        # input_assigned = track_ids == -1
        # used_track_ids = set(track_ids)
        # track_assigned = np.array([0]+[i+1 for i in range(len(pred_prob_with_bg)-1) if i not in used_track_ids])

        # filt_mask = mask[input_assigned]
        # filt_pred = pred_prob_with_bg[track_assigned]

        # assign the unassigned masks
        mask, input_index = assign_masks(
            # filt_pred, filt_mask,
            pred_prob_with_bg, mask, 
            allowance_mask=self.allowance_mask, 
            new_detection_mask=self.new_detection_mask,
            **self.params)
        # track_ids[input_assigned] = input_index
        # input_index = track_ids

        labels = self.labels
        return mask, input_index, labels
    
    def postprocess(self, mask):
        # if self.negative_mask is not None:
        #     mask[:, self.negative_mask > 0.5] = 0
        return mask



def iou_assignment(first_mask, other_mask, min_iou=0.4):
    iou = mask_iou(first_mask, other_mask)
    iou = iou.cpu().numpy() if isinstance(iou, torch.Tensor) else iou
    track_ids, other_ids = linear_sum_assignment(iou, maximize=True)
    if min_iou:
        cost = iou[track_ids, other_ids]
        track_ids = track_ids[cost > min_iou]
        other_ids = other_ids[cost > min_iou]
    return track_ids, other_ids



def assign_masks(
        pred_prob_with_bg, 
        new_masks, 
        label_cost=None, 
        new_detection_mask=None, 
        allowance_mask=None,
        min_iou=0.4, 
        min_box_iou=0.8, 
        min_box_ioa=None, 
        # max_center_dist=0.3, 
        # min_box_center_ioa=0.2, 
        min_label_cost=0.5, 
        **kw
    ):
    '''Assign predicted masks to user-provided masks.

    Arguments:
        pred_prob_with_bg (torch.Tensor): The probabilistic masks given by XMem.
        new_masks (torch.Tensor): The binary masks given by you.
        track_index (np.array): Pre-defined detection -> track index mapping. Should be the length of detections with elements 
            corresponding to track indices. Any values containing -1 will be considered in the assignment.
        new_detection_index (torch.Tensor | None): Indices of detection masks that we should track. This lets us provide additional masks 
            that we may want to compare against, but still not track.
        allowance_mask (np.ndarray): shape [N_tracks, N_input_masks], mask for allowed track-input assignment. Can either be boolean or 
            a scalar representing a weight applied to the cost. If specified, cost = cost + (allowance_mask - 1). 
            Use 0 to disallow matching, 1 for normal assignment, and 2 (or any >1) for prioritized assignment.
        min_iou (float): The minimum IoU allowed for mask assignment.
        min_box_iou (float): The minimum bounding box IoU allowed for mask assignment.
        min_box_ioa (float): The minimum box IoA (intersection over smallest area) allowed for mask assignment. This 
            lets you match boxes that are entirely within another box.
    
    Returns:
        full_masks (torch.Tensor): The merged masks.
        input_track_ids (list): The track indices corresponding to each input mask. If allow_create=False, these values can be None.
        unmatched_tracks (list): Tracks that did not have associated matches.
        new_tracks (torch.Tensor): Track indices that were added.

    NOTE: returned track IDs correspond to the track index in ``binary_mask``. If you have
          another index of tracks (e.g. if you manage track deletions) you need to re-index
          those externally.
    '''
    # get binary and probabilistic masks of xmem predictions
    binary_masks = mask_pred_to_binary(pred_prob_with_bg)[1:]
    pred_masks = pred_prob_with_bg[1:]  # drop background

    # filter out aux detections
    all_masks = new_masks# if other_masks is None else torch.cat([new_masks, other_masks], dim=0)
    if new_detection_mask is not None:
        new_masks = all_masks[new_detection_mask]
    
    # compute segmentation iou cost
    cost = mask_iou(binary_masks, all_masks).cpu().numpy()

    # allow user to forbid certain combinations
    assign_cost = cost
    if allowance_mask is not None:
        # 1: allowed, 0: not allowed, >1: forced assign
        assign_cost = assign_cost + (allowance_mask-1)

    # do the assignment: rows = track_ids, cols = input_ids
    rows, cols = linear_sum_assignment(assign_cost, maximize=True)
    
    # has a high enough segmentation match
    keep = (
        (cost[rows, cols] > min_iou)
    )
    
    # weaker assignment based on box iou/label match
    if min_box_iou is not None or min_box_ioa is not None:
        tboxes = masks_to_boxes2(binary_masks)
        nboxes = masks_to_boxes2(all_masks)
        iou_cost, ioa_cost = box_ioa(tboxes, nboxes)
        weak_keep = np.full_like(keep, False)
        if min_box_iou is not None:
            keep |= iou_cost[rows, cols].cpu().numpy() > min_box_iou
        if min_box_ioa is not None:
            weak_keep |= ioa_cost[rows, cols].cpu().numpy() > min_box_ioa
        if label_cost is not None:
            weak_keep &= label_cost[rows, cols].cpu().numpy() > min_label_cost
        keep |= weak_keep

    if allowance_mask is not None:
        # vote for user-incentivized combinations
        keep |= allowance_mask[rows, cols] > 1
        # drop user-forbidden combinations
        keep &= allowance_mask[rows, cols] > 0

    log.debug('tracks: %d %s', len(pred_masks), rows)
    log.debug('detections: %d %s', len(new_masks), cols)
    log.debug('keep: %s', keep)
    log.debug('cost: %s', cost[rows, cols])
    log.debug('cost: \n%s', np.round(cost, 2))

    # select the kept subset
    rows = rows[keep]
    cols = cols[keep]

    # remap detection index to exclude ignored detections
    if new_detection_mask is not None:
        # c1,r1=cols, rows
        indices = np.where(new_detection_mask)[0]
        index_mapping = np.zeros(len(new_detection_mask), dtype=int)-1
        index_mapping[indices] = np.arange(len(indices))
        cols = index_mapping[cols]
        keep = cols >= 0
        rows = rows[keep]
        cols = cols[keep]

    mapping = dict(zip(cols, rows))
    track_ids = [mapping.get(i) for i in range(len(new_masks))]
    return new_masks, track_ids
    # return new_masks[cols], rows
    # return combine_masks(pred_masks, new_masks, rows, cols, **kw)


def combine_masks(pred_masks, new_masks, rows, cols, allow_create=True, join_method='replace'):
    # existing tracks without a matching detection
    unmatched_rows = sorted(set(range(len(pred_masks))) - set(rows))
    print("Unmatched tracks", unmatched_rows)
    # new detections without a matching track
    unmatched_cols = sorted(set(range(len(new_masks))) - set(cols))
    print("Unmatched detections", unmatched_cols)

    # ---------------------- Merge everything into one mask ---------------------- #

    # create indices for new tracks
    new_rows = torch.arange(len(unmatched_cols) if allow_create else 0) + len(pred_masks)
    # merge masks - create blank array with the right size
    n = len(pred_masks) + len(new_rows)
    full_masks = torch.zeros((n, *pred_masks.shape[1:]), device=pred_masks.get_device())
    new_masks = new_masks.float()

    # # first override matches
    if len(rows):
        if join_method == 'replace':  # trust detection masks
            full_masks[rows] = new_masks[cols]
        elif join_method == 'ignore':  # trust tracking masks
            full_masks[rows] = pred_masks[rows]
        elif join_method == 'max':  # take the maximum of the two masks XXX idk if this makes sense
            full_masks[rows] = torch.maximum(new_masks[cols], pred_masks[rows])
        elif join_method == 'min':  # take the minimum of the two masks XXX idk if this makes sense
            full_masks[rows] = torch.minimum(new_masks[cols], pred_masks[rows])
        elif join_method == 'mult':  # scale the likelihood of XMem using the detections [0.5-1.5]
            full_masks[rows] = (new_masks[cols] + 0.5) * pred_masks[rows]
        else:
            raise ValueError("Invalid mask join method")
    # then for tracks that weren't matched, insert the xmem predictions
    if len(unmatched_rows):
        full_masks[unmatched_rows] = pred_masks[unmatched_rows]
    # for new detections without a track, insert with new track IDs
    if len(new_rows):
        full_masks[new_rows] = new_masks[unmatched_cols]

    # this is the track_ids corresponding to the input masks
    new_rows_list = new_rows.tolist()
    if not allow_create:
        new_rows_list = [None]*len(unmatched_cols)
    input_track_ids = [
        r for c, r in sorted(zip(
            (*cols, *unmatched_cols), 
            (*rows, *new_rows_list)))]
    return full_masks, input_track_ids, unmatched_rows, new_rows



# ---------------------------------------------------------------------------- #
#                                  Comparisons                                 #
# ---------------------------------------------------------------------------- #


def mask_pred_to_binary(x):
    idxs = torch.argmax(x, dim=0)
    y = torch.zeros_like(x)
    for i in range(len(x)):
        y[i, idxs==i] = 1
    return y

def mask_iou(a, b, eps=1e-7):
    a, b = a[:, None], b[None]
    overlap = (a * b) > 0
    union = (a + b) > 0
    return 1. * overlap.sum((2, 3)) / (union.sum((2, 3)) + eps)

# def mask_iou_a_engulf_b(a, b, eps=1e-7):
#     a, b = a[:, None], b[None]
#     overlap = (a * b) > 0
#     return 1. * overlap.sum((2, 3)) / (b.sum((2, 3)) + eps)

def box_ioa(xx, yy, method='union', eps=1e-7):
    # Calculate the area of intersection
    intersection_area = (
        torch.clamp(torch.minimum(xx[:, None, 2], yy[None, :, 2]) - torch.maximum(xx[:, None, 0], yy[None, :, 0]), 0) * 
        torch.clamp(torch.minimum(xx[:, None, 3], yy[None, :, 3]) - torch.maximum(xx[:, None, 1], yy[None, :, 1]), 0)
    )

    # Calculate the area of each bounding box
    area_xx = (xx[:, 2] - xx[:, 0]) * (xx[:, 3] - xx[:, 1])
    area_yy = (yy[:, 2] - yy[:, 0]) * (yy[:, 3] - yy[:, 1])
    # if method == 'union':
    #     base_area = area_xx[:, None] + area_yy[None] - intersection_area
    # elif method == 'min':
    #     base_area = torch.minimum(area_xx[:, None], area_yy[None])
    # elif method == 'max':
    #     base_area = torch.maximum(area_xx[:, None], area_yy[None])
    # else:
    #     raise ValueError(f"Invalid box IoU method: {method}")
    union_area = area_xx[:, None] + area_yy[None] - intersection_area
    min_area = torch.minimum(area_xx[:, None], area_yy[None])

    # Calculate IoU (Intersection over Area)
    return intersection_area / (union_area + eps), intersection_area / (min_area + eps)

def box_center_dist(xx, yy, eps=1e-7):
    # Calculate the center coordinates of each bounding box
    center_xx = torch.stack([(xx[:, 0] + xx[:, 2]) / 2, (xx[:, 1] + xx[:, 3]) / 2], dim=1)
    center_yy = torch.stack([(yy[:, 0] + yy[:, 2]) / 2, (yy[:, 1] + yy[:, 3]) / 2], dim=1)
    center_distance = torch.norm(center_xx[:, None, :] - center_yy[None, :, :], dim=2)

    # Calculate the minimum width and height for each pair of boxes
    width = torch.maximum(xx[:, None, 2] - xx[:, None, 0], yy[:, 2] - yy[:, 0])
    height = torch.maximum(xx[:, None, 3] - xx[:, None, 1], yy[:, 3] - yy[:, 1])
    base_dist = torch.minimum(width, height)

    # Calculate box center distance divided by the minimum width/height
    return center_distance / (base_dist + eps)


# ---------------------------------------------------------------------------- #
#                                     Utils                                    #
# ---------------------------------------------------------------------------- #


def masks_to_boxes2(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    bounding_boxes = torch.zeros((masks.shape[0], 4), device=masks.device, dtype=torch.float)
    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        if len(x):
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)
    return bounding_boxes
