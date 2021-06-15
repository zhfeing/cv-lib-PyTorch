from typing import List

from torch import Tensor, LongTensor, BoolTensor

from .ap_meter import APMeter_Base


class APMeter_COCO(APMeter_Base):
    """
    COCO ap meter
    """

    def __init__(self, num_classes: int):
        super().__init__(num_classes=num_classes)

    def update(
        self,
        img_ids: List[LongTensor],
        pred_bboxes: List[Tensor],
        pred_labels: List[Tensor],
        pred_scores: List[Tensor],
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
        gt_difficult: List[BoolTensor]
    ):
        super().update(
            img_ids,
            pred_bboxes,
            pred_labels,
            pred_scores,
            gt_bboxes,
            gt_labels,
            gt_difficult
        )
