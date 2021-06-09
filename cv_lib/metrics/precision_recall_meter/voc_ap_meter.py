from typing import List
from collections import defaultdict

import tqdm

import torch
from torch import Tensor
from torchvision.ops.boxes import box_iou

import numpy as np

from ..meter import Meter


class VOCPrecisionRecallMeter(Meter):
    """
    Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of predicted bounding
    boxes obtained from a dataset which has :math:`N` images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes: An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        pred_labels: An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores: An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes: An iterable of ground truth bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels: An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults: An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
    """

    def __init__(self, iou_threshold: float = 0.5, use_07_metric: bool = False):
        self.iou_threshold = iou_threshold
        self.use_07_metric = use_07_metric

        self.pred_bboxes: List[Tensor] = list()
        self.pred_labels: List[Tensor] = list()
        self.pred_scores: List[Tensor] = list()
        self.gt_bboxes: List[Tensor] = list()
        self.gt_labels: List[Tensor] = list()
        self.gt_difficult: List[Tensor] = list()

    def reset(self):
        self.pred_bboxes.clear()
        self.pred_labels.clear()
        self.pred_scores.clear()
        self.gt_bboxes.clear()
        self.gt_labels.clear()
        self.gt_difficult.clear()

    def update(
        self,
        pred_bboxes: List[Tensor],
        pred_labels: List[Tensor],
        pred_scores: List[Tensor],
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
        gt_difficult: List[Tensor]
    ):
        """
        Update a batch of predicts. For each img, all tensors must be located on the same device
        """
        assert len(pred_bboxes) == len(pred_labels) == len(pred_scores) == len(gt_bboxes) == len(gt_labels) == len(gt_difficult)
        self.pred_bboxes.extend(pred_bboxes)
        self.pred_labels.extend(pred_labels)
        self.pred_scores.extend(pred_scores)
        self.gt_bboxes.extend(gt_bboxes)
        self.gt_labels.extend(gt_labels)
        self.gt_difficult.extend(gt_difficult)

    def prec_rec(self):
        n_pos = defaultdict(int)
        score = defaultdict(list)
        match = defaultdict(list)
        batch_iter = zip(
            self.pred_bboxes,
            self.pred_labels,
            self.pred_scores,
            self.gt_bboxes,
            self.gt_labels,
            self.gt_difficult
        )
        # for each image
        for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in tqdm.tqdm(batch_iter, total=len(self.pred_bboxes), desc="Image"):
            # convert difficult
            if gt_difficult is None:
                gt_difficult = torch.zeros(gt_bbox.shape[0], device=gt_bbox.device, dtype=torch.bool)
            # for each label
            for label in torch.unique(torch.cat((pred_label, gt_label))):
                label = label.item()
                pred_mask_l = pred_label == label
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort(descending=True)
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == label
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                n_pos[label] += torch.sum(~gt_difficult_l).item()
                score[label].extend(pred_score_l.tolist())

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    match[label].extend((0,) * pred_bbox_l.shape[0])
                    continue

                iou = box_iou(pred_bbox_l, gt_bbox_l)
                gt_iou, gt_index = iou.max(axis=1)
                # set -1 if there is no matching ground truth
                gt_index[gt_iou < self.iou_threshold] = -1

                selec = torch.zeros(gt_bbox_l.shape[0], device=gt_bbox.device, dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            match[label].append(-1)
                        else:
                            if not selec[gt_idx]:
                                match[label].append(1)
                            else:
                                match[label].append(0)
                        selec[gt_idx] = True
                    else:
                        match[label].append(0)

        n_fg_class = max(n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for label in n_pos.keys():
            score_l = torch.tensor(score[label])
            match_l = torch.tensor(match[label], dtype=torch.int8)

            order = score_l.argsort(descending=True)
            match_l = match_l[order]
            tp = torch.cumsum(match_l == 1, dim=0)
            fp = torch.cumsum(match_l == 0, dim=0)

            # If an element of fp + tp is 0, the corresponding element of prec[l] is nan.
            prec[label] = tp / (fp + tp)
            # If n_pos[l] is 0, rec[l] is None.
            if n_pos[label] > 0:
                rec[label] = tp / n_pos[label]

        return prec, rec

    def ap(self, prec: List[np.ndarray], rec: List[np.ndarray], use_07_metric: bool = False):
        """Calculate average precisions based on evaluation code of PASCAL VOC.

        This function calculates average precisions
        from given precisions and recalls.
        The code is based on the evaluation code used in PASCAL VOC Challenge.

        Args:
            prec (list of numpy.array): A list of arrays.
                :obj:`prec[l]` indicates precision for class :math:`l`.
                If :obj:`prec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            rec (list of numpy.array): A list of arrays.
                :obj:`rec[l]` indicates recall for class :math:`l`.
                If :obj:`rec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
                for calculating average precision. The default value is
                :obj:`False`.

        Returns:
            ~numpy.ndarray:
            This function returns an array of average precisions.
            The :math:`l`-th value corresponds to the average precision
            for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
            :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

        """

        n_fg_class = len(prec)
        ap = np.empty(n_fg_class)
        for label in range(n_fg_class):
            if prec[label] is None or rec[label] is None:
                ap[label] = np.nan
                continue

            if use_07_metric:
                # 11 point metric
                ap[label] = 0
                for t in np.arange(0., 1.1, 0.1):
                    if np.sum(rec[label] >= t) == 0:
                        p = 0
                    else:
                        p = np.max(np.nan_to_num(prec[label])[rec[label] >= t])
                    ap[label] += p / 11
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mpre = np.concatenate(([0], np.nan_to_num(prec[label]), [0]))
                mrec = np.concatenate(([0], rec[label], [1]))

                mpre = np.maximum.accumulate(mpre[::-1])[::-1]

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap[label] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def value(self):
        """
        Return:
            "ap"
            "map"
            "prec"
            "rec"
        """
        prec, rec = self.prec_rec()
        prec = list(x.cpu().numpy() if isinstance(x, Tensor) else x for x in prec)
        rec = list(x.cpu().numpy() if isinstance(x, Tensor) else x for x in rec)
        ap = self.ap(prec, rec, self.use_07_metric)
        return {"ap": ap.tolist(), "map": np.nanmean(ap), "prec": prec, "rec": rec}
