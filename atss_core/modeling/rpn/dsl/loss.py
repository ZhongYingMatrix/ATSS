import torch
import math
from torch import nn
import torch.nn.functional as F
import os
from atss_core.layers import SigmoidFocalLoss


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class DSLLossComputation(object):
    def __init__(self, cfg):
        self.topk = cfg.MODEL.DSL.TOPK
        self.strides = cfg.MODEL.DSL.FPN_STRIDES
        self.cls_out_channels = cfg.MODEL.DSL.NUM_CLASSES - 1
        self.reg_loss_weight = cfg.MODEL.DSL.REG_LOSS_WEIGHT
        self.loss_centerness = nn.BCEWithLogitsLoss(reduction="sum")
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.DSL.LOSS_GAMMA,
                                              cfg.MODEL.DSL.LOSS_ALPHA)

    def __call__(self, cls_scores, bbox_preds, centernesses,
                 targets, all_level_points):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        gt_bboxes = [target_im.bbox for target_im in targets]
        gt_labels = [target_im.get_field('labels') for target_im in targets]
        candidate_idxs = self.get_candidate_idxs(gt_bboxes, all_level_points)

        loss_bbox, num_pos, num_ctr_pos = 0, 0, 0
        cls_targets = [torch.zeros_like(cls_score) for cls_score in cls_scores]
        num_imgs = cls_scores[0].size(0)
        ctr_pred_lst, ctr_target_lst = [], []

        for img_id in range(num_imgs):
            cls_lst = []
            de_bbox_lst = []
            centerpoint_lst = []
            ctrness_lst = []
            for lvl_id in range(len(self.strides)):
                flatten_cls_img_lvl = cls_scores[lvl_id][img_id].permute(
                    1, 2, 0).reshape(-1, self.cls_out_channels)
                flatten_bbox_img_lvl = bbox_preds[lvl_id][img_id].permute(
                    1, 2, 0).reshape(-1, 4)
                flatten_ctrness_img_lvl = centernesses[lvl_id][img_id].permute(
                    1, 2, 0).reshape(-1, 1)
                candidate_cls_img_lvl = flatten_cls_img_lvl[
                    candidate_idxs[img_id][lvl_id], :]
                candidate_bbox_img_lvl = flatten_bbox_img_lvl[
                    candidate_idxs[img_id][lvl_id], :]
                candidate_ctrness_img_lvl = flatten_ctrness_img_lvl[
                    candidate_idxs[img_id][lvl_id], :]
                candidate_points_img_lvl = all_level_points[lvl_id][
                    candidate_idxs[img_id][lvl_id], :]
                candidate_decoded_bbox_img_lvl = self.distance2bbox(
                    candidate_points_img_lvl.reshape(-1, 2),
                    candidate_bbox_img_lvl.reshape(-1, 4)).reshape(
                        self.topk, -1, 4)
                cls_lst.append(candidate_cls_img_lvl)
                de_bbox_lst.append(candidate_decoded_bbox_img_lvl)
                centerpoint_lst.append(candidate_points_img_lvl)
                ctrness_lst.append(candidate_ctrness_img_lvl)
            cls_lst_gt = [
                torch.cat([
                    cls_lst[lvl_id][:, gt_id, gt_labels[img_id][gt_id] - 1]
                    for lvl_id in range(len(self.strides))
                ], dim=0) for gt_id in range(len(gt_bboxes[img_id]))
            ]
            de_bbox_gt = [
                torch.cat([
                    de_bbox_lst[lvl_id][:, gt_id, :]
                    for lvl_id in range(len(self.strides))
                ], dim=0) for gt_id in range(len(gt_bboxes[img_id]))
            ]
            centerpoint_gt = [
                torch.cat([
                    centerpoint_lst[lvl_id][:, gt_id, :]
                    for lvl_id in range(len(self.strides))
                ], dim=0) for gt_id in range(len(gt_bboxes[img_id]))
            ]
            ctrness_gt = [
                torch.cat([
                    ctrness_lst[lvl_id][:, gt_id, :]
                    for lvl_id in range(len(self.strides))
                ], dim=0) for gt_id in range(len(gt_bboxes[img_id]))
            ]
            cls_lst_gt = torch.cat(
                [cls_gt[None, ...] for cls_gt in cls_lst_gt])
            de_bbox_gt = torch.cat([db_gt[None, ...] for db_gt in de_bbox_gt])
            centerpoint_gt = torch.cat(
                [cp_gt[None, ...] for cp_gt in centerpoint_gt])
            ctrness_gt = torch.cat(
                [cn_gt[None, ...] for cn_gt in ctrness_gt]).squeeze(dim=2)

            gt_bboxes_img = gt_bboxes[img_id][:, None, :].repeat(
                1,
                len(self.strides) * self.topk, 1)
            _de_bbox_gt = self.CIoU(
                de_bbox_gt.reshape(-1, 4),
                gt_bboxes_img.reshape(-1, 4)).reshape(
                    -1,
                    len(self.strides) * self.topk)
            de_bbox_gt = _de_bbox_gt.clamp(min=1e-6)

            with torch.no_grad():
                scores = de_bbox_gt * cls_lst_gt.sigmoid()
                threshold = (scores.mean(dim=1) + scores.std(dim=1))
                threshold = threshold[:, None].repeat(
                    1,
                    len(self.strides) * self.topk)
                keep_idxmask = (scores >= threshold)
                inside_gt_bbox_mask = (
                    (centerpoint_gt[..., 0] > gt_bboxes_img[..., 0])
                    * (centerpoint_gt[..., 0] < gt_bboxes_img[..., 2])
                    * (centerpoint_gt[..., 1] > gt_bboxes_img[..., 1])
                    * (centerpoint_gt[..., 1] < gt_bboxes_img[..., 3]))
                keep_idxmask *= inside_gt_bbox_mask

            center_p = centerpoint_gt.view(-1, 2)
            gt = gt_bboxes_img.view(-1, 4)
            left = center_p[:, 0] - gt[:, 0]
            right = gt[:, 2] - center_p[:, 0]
            up = center_p[:, 1] - gt[:, 1]
            down = gt[:, 3] - center_p[:, 1]
            l_r = torch.stack((left, right)).clamp(min=1e-6)
            u_d = torch.stack((up, down)).clamp(min=1e-6)
            centerness = ((l_r.min(dim=0)[0] * u_d.min(dim=0)[0])
                          / (l_r.max(dim=0)[0] * u_d.max(dim=0)[0])).sqrt()
            centerness = centerness.reshape(
                -1,
                len(self.strides) * self.topk)[keep_idxmask]
            ctrness_pred = ctrness_gt[keep_idxmask]
            ctr_target_lst.append(centerness)
            ctr_pred_lst.append(ctrness_pred)
            reweight_factor = centerness
            num_pos += reweight_factor.sum()
            num_ctr_pos += keep_idxmask.sum()
            loss_bbox += ((1 - _de_bbox_gt[keep_idxmask])
                          * reweight_factor).sum()

            with torch.no_grad():
                soft_label = de_bbox_gt / de_bbox_gt.max(
                    dim=1)[0][:, None].repeat(
                        1,
                        len(self.strides) * self.topk).detach()
                soft_label = soft_label.permute(1, 0).reshape(
                    len(self.strides), self.topk, -1)
            keep_idxmask = keep_idxmask.permute(1, 0).reshape(
                len(self.strides), self.topk, -1)
            for lvl_id in range(len(self.strides)):
                keep_idxmask_lvl = keep_idxmask[lvl_id]
                candidate_idxs_lvl = candidate_idxs[img_id][lvl_id][
                    keep_idxmask_lvl]
                gt_labels_lvl = gt_labels[img_id][None, :].repeat(
                    self.topk, 1)[keep_idxmask_lvl] - 1
                soft_label_lvl = soft_label[lvl_id][keep_idxmask_lvl]
                cls_targets[lvl_id][img_id].view(
                    self.cls_out_channels,
                    -1)[gt_labels_lvl,
                        candidate_idxs_lvl] = soft_label_lvl

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(num_pos).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        total_num_ctr_pos = reduce_sum(num_ctr_pos).item()
        num_ctr_pos_avg_per_gpu = max(total_num_ctr_pos / float(num_gpus), 1.0)
        loss_centerness = self.loss_centerness(
            torch.cat(ctr_pred_lst), torch.cat(ctr_target_lst))
        loss_centerness /= num_ctr_pos_avg_per_gpu
        loss_bbox /= num_pos_avg_per_gpu
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_targets = [
            cls_target.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_target in cls_targets
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_cls_targets = torch.cat(flatten_cls_targets)
        loss_cls = self.py_sigmoid_focal_loss(
            flatten_cls_scores, flatten_cls_targets).sum() / num_pos_avg_per_gpu

        return loss_cls, loss_bbox * self.reg_loss_weight, loss_centerness

    def get_candidate_idxs(self, gt_bboxes, all_level_points):
        # img * [lvl * idxs(9*nums_gt)]
        candidate_idxs = []
        for gt_b in gt_bboxes:
            candidate_idxs_img = []
            gt_cx = (gt_b[:, 0] + gt_b[:, 2]) / 2.0
            gt_cy = (gt_b[:, 1] + gt_b[:, 3]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)
            for lvl_points in all_level_points:
                distances = (lvl_points[:, None, :]
                             - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                _, topk_idxs = distances.topk(self.topk, dim=0, largest=False)
                candidate_idxs_img.append(topk_idxs)
            candidate_idxs_img = torch.cat([idxs[None, ...]
                                            for idxs in candidate_idxs_img])
            candidate_idxs.append(candidate_idxs_img)
        return candidate_idxs

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1] - 1)
            y1 = y1.clamp(min=0, max=max_shape[0] - 1)
            x2 = x2.clamp(min=0, max=max_shape[1] - 1)
            y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        return torch.stack([x1, y1, x2, y2], -1)

    def CIoU(self, pred, target, eps=1e-7):
        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
        ag = (target[:, 2] - target[:, 0] + 1) * (
            target[:, 3] - target[:, 1] + 1)
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose diag
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
        enclose_diag = (enclose_wh[:, 0].pow(2)
                        + enclose_wh[:, 1].pow(2) + eps)

        # center distance
        xp = (pred[:, 0] + pred[:, 2]) / 2
        yp = (pred[:, 1] + pred[:, 3]) / 2
        xg = (target[:, 0] + target[:, 2]) / 2
        yg = (target[:, 1] + target[:, 3]) / 2
        center_d = (xp - xg).pow(2) + (yp - yg).pow(2)

        # DIoU
        dious = ious - center_d / enclose_diag

        # CIoU
        w_gt = target[:, 2] - target[:, 0] + 1
        h_gt = target[:, 3] - target[:, 1] + 1
        w_pred = pred[:, 2] - pred[:, 0] + 1
        h_pred = pred[:, 3] - pred[:, 1] + 1
        v = (4 / math.pi ** 2) * torch.pow(
            (torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
        with torch.no_grad():
            S = 1 - ious
            alpha = v / (S + v)
        cious = dious - alpha * v

        return cious

    def py_sigmoid_focal_loss(self,
                              pred,
                              target,
                              weight=None,
                              gamma=2.0,
                              alpha=0.25):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha)
                        * (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        return loss


def make_dsl_loss_evaluator(cfg):
    loss_evaluator = DSLLossComputation(cfg)
    return loss_evaluator
