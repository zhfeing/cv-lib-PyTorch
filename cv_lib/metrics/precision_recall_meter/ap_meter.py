import collections
import itertools
from typing import Iterable, List, Dict, Any
import math

import numpy as np

import torch
from torch import Tensor, BoolTensor, LongTensor
from torchvision.ops.boxes import box_iou

from cv_lib.metrics.meter import Meter
from cv_lib.utils.cuda_utils import list_to_device
from cv_lib.utils.basic_utils import customized_argsort
import cv_lib.distributed.utils as dist_utils


class APMeter_Base(Meter):
    """
    COCO-like ap meter. Support evaluating with multiple gpus.

    Note:
        1. All class must be in [1, ..., `num_classes`]
        2. All bounding boxes must be in form `xyxy` with or without normalizing by image width and height
        3. All img id must be `LongTensor` type
    """

    def __init__(
        self,
        num_classes: int,
        max_det: int = 100,
        iou_thresholds: List[float] = None,
        recall_steps: List[float] = None
    ):
        """
        Args:
            num_classes: including background (`0`)
            max_det: max detection number for every image and label combination
        """
        self.max_det = max_det
        # set thresholds
        if iou_thresholds is None:
            self.iou_thresholds = torch.linspace(0.5, 0.95, steps=10)
        else:
            self.iou_thresholds = torch.tensor(iou_thresholds, dtype=torch.float)
        if recall_steps is None:
            self.recall_steps = torch.linspace(0, 1, steps=101)
        else:
            self.recall_steps = torch.tensor(recall_steps, dtype=torch.float)
        # ignore background `0`
        self.label_ids = torch.arange(1, num_classes)
        self.img_ids: List[int] = list()
        self.eval_results: List[np.ndarray] = list()
        self.accumulate_info: Dict[str, Any] = None

    def reset(self):
        self.img_ids.clear()
        self.eval_results.clear()
        self.accumulate_info = None

    def update(
        self,
        img_ids: List[LongTensor],
        pred_boxes: List[Tensor],
        pred_labels: List[Tensor],
        pred_scores: List[Tensor],
        gt_boxes: List[Tensor],
        gt_labels: List[Tensor],
        gt_hards: List[BoolTensor]
    ):
        """
        Update a batch of predicts. For each img, all tensors must be located on the same device
        """
        kwargs = dict(
            img_ids=img_ids,
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_hards=gt_hards
        )
        # move to cpu first
        kwargs = {k: list_to_device(v, "cpu") for k, v in kwargs.items()}
        # get predictions and ground truths
        prs, gts = self._prepare(kwargs)
        # calculate iou between pr and gt
        ious = self._cal_iou(prs, gts, img_ids)
        # batch evaluation
        res_batch = self._eval_batch(img_ids, prs, gts, ious)
        # extend batch info
        self.img_ids.extend(list(i.item() for i in img_ids))
        self.eval_results.append(res_batch)

    def _prepare(self, kwargs: Dict[str, List[Tensor]]):
        """
        Get splitted predictions and ground truths by dictionary with form:
        1. Predictions:
            {
                id: int with range [0--bs*N)
                box: Tensor with shape [4]
                label: LongTensor
                score: Tensor
            }
        2. Ground truths
            {
                id: int with range [0--num_gt)
                box: Tensor with shape [4]
                label: LongTensor
                hard: BoolTensor
                ignore: BoolTensors
            }

        Return:
            defaultdict of pr dict with key (image_id, label) from this batch
            defaultdict of gt dict with key (image_id, label) from this batch
        """
        gt_ignores: List[BoolTensor] = list(hard.clone() for hard in kwargs["gt_hards"])
        kwargs["gt_ignores"] = gt_ignores
        # expand img ids
        pr_bs = list(b.shape[0] for b in kwargs["pred_boxes"])
        gt_bs = list(b.shape[0] for b in kwargs["gt_boxes"])
        img_ids = kwargs.pop("img_ids")
        kwargs["pr_img_ids"] = expand_img_id(img_ids, pr_bs)
        kwargs["gt_img_ids"] = expand_img_id(img_ids, gt_bs)

        # concat
        for k, v in kwargs.items():
            kwargs[k] = torch.cat(v)
        # split all predictions
        prs: Dict[tuple, List[Dict[str, Tensor]]] = collections.defaultdict(list)
        # split predictions
        pr_iter: Iterable[List[Tensor]] = zip(
            kwargs["pr_img_ids"],
            kwargs["pred_boxes"],
            kwargs["pred_labels"],
            kwargs["pred_scores"]
        )
        for pr_id, (image_id, pr_box, pr_label, pr_score) in enumerate(pr_iter):
            pr = dict(
                id=pr_id,
                box=pr_box,
                label=pr_label,
                score=pr_score,
            )
            prs[(image_id.item(), pr_label.item())].append(pr)
        # split ground truths
        gts: Dict[tuple, List[str, Tensor]] = collections.defaultdict(list)
        gt_iter: Iterable[List[Tensor]] = zip(
            kwargs["gt_img_ids"],
            kwargs["gt_boxes"],
            kwargs["gt_labels"],
            kwargs["gt_hards"],
            kwargs["gt_ignores"]
        )
        for gt_id, (image_id, gt_box, gt_label, gt_hard, gt_ignore) in enumerate(gt_iter):
            gt = dict(
                id=gt_id,
                box=gt_box,
                label=gt_label,
                hard=gt_hard,
                ignore=gt_ignore
            )
            gts[(image_id.item(), gt_label.item())].append(gt)
        return prs, gts

    def _cal_iou(
        self,
        prs: Dict[tuple, List[Dict[str, Tensor]]],
        gts: Dict[tuple, List[Dict[str, Tensor]]],
        img_ids: List[LongTensor]
    ) -> Dict[tuple, Tensor]:
        """
        IOU of predictions and ground truths.
        Note:
            1. predict bounding box are sorted by score in this function, and removed tail by `max_det`
            2. predictions will not be changed
        """
        ious: Dict[tuple, Tensor] = dict()
        for img_id, label in itertools.product(img_ids, self.label_ids):
            idx = (img_id.item(), label.item())
            pr = prs.get(idx, list())
            gt = gts.get(idx, list())
            if len(pr) > 0 and len(gt) > 0:
                pr = prs[idx]
                gt = gts[idx]
                gt_boxes = list(p["box"] for p in gt)
                gt_boxes = torch.stack(gt_boxes)
                # sort predictions
                pr_boxes = list(p["box"] for p in pr)
                pr_scores = list(-p["score"] for p in pr)
                # stable sort
                pr_sort_idx = np.argsort(pr_scores, kind="mergesort")
                pr_boxes = list(pr_boxes[i] for i in pr_sort_idx[:self.max_det])
                pr_boxes = torch.stack(pr_boxes)
                iou = box_iou(pr_boxes, gt_boxes)
                ious[idx] = iou
            else:
                ious[idx] = torch.empty(len(pr), len(gt))
        return ious

    def _eval_batch(
        self,
        img_ids: List[LongTensor],
        prs: Dict[tuple, List[Dict[str, Tensor]]],
        gts: Dict[tuple, List[Dict[str, Tensor]]],
        ious: Dict[tuple, Tensor]
    ):
        batch_res = list()
        for label, img_id in itertools.product(self.label_ids, img_ids):
            idx = (img_id.item(), label.item())
            pr = prs.get(idx, list())
            gt = gts.get(idx, list())
            res = self._eval_img(pr, gt, ious[idx])
            if res is not None:
                info = dict(
                    image_id=img_id,
                    label=label
                )
                res.update(info)
            batch_res.append(res)
        batch_res = np.asarray(batch_res).reshape(self.label_ids.shape[0], len(img_ids))
        return batch_res

    def _eval_img(
        self,
        pr: List[Dict[str, Tensor]],
        gt: List[Dict[str, Tensor]],
        iou: Tensor
    ) -> Dict[str, Any]:
        """
        Return: dict with keys
            {
                pr_scores:Tensor with shape [N_pr]
                gt_ignore: BoolTensor with shape [N_gt]
                pr_ignore: BoolTensor with shape [N_iou_thrs, N_pr]
                pr_match: LongTensor with shape [N_iou_thrs, N_pr]
                gt_match: LongTensor with shape [N_iou_thrs, N_gt]
            }
        """
        n_pr = len(pr)
        n_gt = len(gt)
        n_iou_thrs = self.iou_thresholds.shape[0]
        if n_pr == 0 and n_gt == 0:
            return None
        gt_ignore = list(g["ignore"] for g in gt)
        # sort ground truth so that the ignored terms fall behind
        gt_sort_idx = np.argsort(gt_ignore, kind="mergesort").tolist()
        gt = list(gt[i] for i in gt_sort_idx)
        # sort predictions by scores
        pr_sort_idx = np.argsort(list(-p["score"] for p in pr), kind="mergesort").tolist()
        pr = list(pr[i] for i in pr_sort_idx[:self.max_det])
        # get other infomation
        is_hard = list(g["hard"] for g in gt)
        # sort iou by teacher idx
        iou = iou[:, gt_sort_idx]

        # ground truth and prediction match matrix
        gt_match = torch.empty(n_iou_thrs, n_gt, dtype=torch.long).fill_(-1)
        pr_match = torch.empty(n_iou_thrs, n_pr, dtype=torch.long).fill_(-1)
        gt_ignore = torch.tensor(gt_ignore, dtype=torch.bool)
        pr_ignore = torch.zeros_like(pr_match, dtype=torch.bool)

        # match between ground truths and predictions
        if n_gt != 0 and n_pr != 0:
            for t_id, iou_thrs in enumerate(self.iou_thresholds):
                for pr_id in range(n_pr):
                    # match with given iou threshold (best )
                    iou_best = min(iou_thrs, 0.9999)
                    # matched gt id
                    match_id = -1
                    for gt_id in range(n_gt):
                        # already matched and current gt is not a hard one, skip this gt
                        if gt_match[t_id, gt_id] > -1 and not is_hard[gt_id]:
                            continue
                        """
                        When current gt are ignored, which means all gt next are ignored for the reason
                        that gt are sorted by ignore. If we have a match with not ignored gt, it means
                        this is the best gt matched and current prediction has been perfectly matched.
                        """
                        if match_id > -1 and not gt_ignore[match_id] and gt_ignore[gt_id]:
                            # jump to next prediction
                            break
                        current_iou = iou[pr_id, gt_id]
                        # current gt is not better than threshold or previous match result
                        if current_iou < iou_best:
                            continue
                        # current gt is better than threshold and previous match result
                        # set new threshold
                        iou_best = current_iou
                        match_id = gt_id
                    # no gt match current prediction
                    if match_id == -1:
                        continue
                    # if matched gt is ignored, set prediction to be ignore
                    pr_ignore[t_id, pr_id] = gt_ignore[match_id]
                    pr_match[t_id, pr_id] = gt[match_id]["id"]
                    gt_match[t_id, match_id] = pr[pr_id]["id"]

        pr_scores = [p["score"].item() for p in pr]
        res = dict(
            pr_scores=torch.as_tensor(pr_scores, dtype=torch.float),
            gt_ignore=gt_ignore,
            pr_ignore=pr_ignore,
            pr_match=pr_match,
            gt_match=gt_match
        )
        return res

    def sync(self):
        self.eval_results = dist_utils.all_gather_list(self.eval_results)

    def accumulate(self):
        # accumulate eval results
        eval_results = np.concatenate(self.eval_results, axis=-1).tolist()
        # get basic numbers
        n_iou = self.iou_thresholds.shape[0]
        n_rec = self.recall_steps.shape[0]
        n_label = self.label_ids.shape[0]

        precision = torch.empty(n_iou, n_rec, n_label, dtype=torch.float).fill_(math.nan)
        scores = torch.empty(n_iou, n_rec, n_label, dtype=torch.float).fill_(math.nan)
        recall = torch.empty(n_iou, n_label, dtype=torch.float).fill_(math.nan)

        for label_id in range(n_label):
            res_by_label = eval_results[label_id]
            res_by_label = list(r for r in res_by_label if r is not None)
            # no record in this category, ignoreing
            if len(res_by_label) == 0:
                continue
            # scores of all predictions in this category
            cat_scores = torch.cat([res["pr_scores"] for res in res_by_label])
            pr_match = torch.cat([res["pr_match"] for res in res_by_label], dim=1)
            gt_ignores = torch.cat([res["gt_ignore"] for res in res_by_label])
            pr_ignores = torch.cat([res["pr_ignore"] for res in res_by_label], dim=1)
            # sort by score
            score_idx = customized_argsort(cat_scores, descending=True, kind="mergesort")
            cat_scores = cat_scores[score_idx]
            pr_match = pr_match[:, score_idx]
            pr_ignores = pr_ignores[:, score_idx]

            n_gt = torch.sum(~gt_ignores)
            # no positive gt, skip this class
            if n_gt == 0:
                continue

            n_pred = cat_scores.shape[0]

            matched = pr_match != -1
            pr_not_ignores = ~pr_ignores
            tps = matched & pr_not_ignores
            fps = ~matched & pr_not_ignores
            tp_sum = torch.cumsum(tps, dim=1, dtype=torch.float)
            fp_sum = torch.cumsum(fps, dim=1, dtype=torch.float)

            cat_rc = tp_sum / n_gt
            cat_pr = tp_sum / (fp_sum + tp_sum)
            # set `nan` to zero
            cat_pr.nan_to_num_()
            if n_pred == 0:
                recall[:, label_id] = 0
            else:
                recall[:, label_id] = cat_rc[:, -1]

            # smooth out the zigzag precision curve
            cat_pr = cat_pr.flip(-1).cummax(-1)[0].flip(-1)
            # precision and score by recall steps [0:0.01:1]
            pr_rc = torch.zeros(n_iou, n_rec)
            score_rc = torch.zeros(n_iou, n_rec)
            for iou_id in range(n_iou):
                recall_steps = torch.searchsorted(cat_rc[iou_id], self.recall_steps, right=False)
                recall_steps = recall_steps[recall_steps < n_pred]
                # fill precision and scores value by step given by recall (pr-rc curve)
                pr = cat_pr[iou_id, recall_steps]
                s = cat_scores[recall_steps]
                perm = torch.arange(0, recall_steps.shape[0], 1)
                pr_rc[iou_id, perm] = pr
                score_rc[iou_id, perm] = s
            precision[..., label_id] = pr_rc
            scores[..., label_id] = score_rc

        self.accumulate_info = dict(
            recall=recall,
            precision=precision,
            scores=scores,
            n_iou=n_iou,
            n_rec=n_rec,
            n_label=n_label,
        )

    def value(self):
        """Get the value of the meter in the current state."""
        assert self.accumulate_info is not None, "Must be accumulated first"
        precision: Tensor = self.accumulate_info["precision"]
        recall: Tensor = self.accumulate_info["recall"]

        average_precision = precision.mean(dim=1)
        ap_50 = average_precision[0]
        ap_75 = average_precision[5]
        ap = average_precision.mean(dim=0)
        ret: Dict[str, Tensor] = dict(
            recall=recall,
            precision=precision,
            average_precision=average_precision,
            ap_50=ap_50,
            ap_75=ap_75,
            ap=ap,
            mean_ap_50=ap_50[~ap_50.isnan()].mean(),
            mean_ap_75=ap_75[~ap_50.isnan()].mean(),
            mean_ap=ap[~ap_50.isnan()].mean(),
            mean_recall=recall[~recall.isnan()].mean(),
        )
        return ret


def expand_img_id(img_ids: List[LongTensor], batch_sizes: List[int]):
    cat_img_ids = list()
    for img_id, bs in zip(img_ids, batch_sizes):
        assert img_id.dim() == 0, "img id must be zero-dim long tensor"
        img_id = img_id.repeat(bs)
        cat_img_ids.append(img_id)
    return cat_img_ids

