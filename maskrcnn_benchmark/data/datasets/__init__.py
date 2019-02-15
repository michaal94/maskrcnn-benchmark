# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .clevr_mini import CLEVR_mini_segmentation
from .clevr_segmentation import CLEVR_segmentation_test
__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "CLEVR_mini_segmentation", "CLEVR_segmentation_test"]
