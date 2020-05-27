import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_dsl_postprocessor
from .loss import make_dsl_loss_evaluator

from atss_core.layers import Scale
from atss_core.layers import DFConv2d


class DSLHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(DSLHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.DSL.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.DSL.FPN_STRIDES

        cls_tower = []
        bbox_tower = []

        for i in range(cfg.MODEL.DSL.NUM_CONVS):
            if self.cfg.MODEL.DSL.USE_DCN_IN_TOWER and \
                    i == cfg.MODEL.DSL.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DSL.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        # TODO necessary? Seem to mimic 8s
        torch.nn.init.constant_(self.bbox_pred.bias, 4)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for i, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[i](self.bbox_pred(box_tower))
            bbox_pred = F.relu(bbox_pred)
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(box_tower))
        return logits, bbox_reg, centerness


class DSLModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(DSLModule, self).__init__()
        self.cfg = cfg
        self.head = DSLHead(cfg, in_channels)
        self.loss_evaluator = make_dsl_loss_evaluator(cfg)
        self.box_selector_test = make_dsl_postprocessor(cfg)
        self.fpn_strides = cfg.MODEL.DSL.FPN_STRIDES

    def forward(self, images, features, targets=None):
        box_cls, box_regression, centerness = self.head(features)
        featmap_sizes = [featmap.size()[-2:] for featmap in box_cls]
        all_level_points = self.get_points(featmap_sizes,
                                           box_regression[0].dtype,
                                           box_regression[0].device)
        img_sizes = [(i[1], i[0]) for i in images.image_sizes]

        if self.training:
            return self._forward_train(box_cls, box_regression,
                                       centerness, targets, all_level_points)
        else:
            return self._forward_test(box_cls, box_regression,
                                      centerness, all_level_points, img_sizes)

    def _forward_train(self, box_cls, box_regression,
                       centerness, targets, all_level_points):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, all_level_points
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, box_cls, box_regression,
                      centerness, all_level_points, img_sizes):
        boxes = self.box_selector_test(box_cls, box_regression,
                                       centerness, all_level_points, img_sizes)
        return boxes, {}

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.fpn_strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points


def build_dsl(cfg, in_channels):
    return DSLModule(cfg, in_channels)
