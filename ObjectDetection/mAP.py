#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/15 下午8:15
# @Author  : jyl
# @File    : mAP.py
import numpy as np
from collections import defaultdict


class mAP:

    def __init__(self, predict, ground_truth, iou_threshold=0.5):
        """
        :param predict: [batch_size, ]
            目标检测算法的输出(已经经过NMS等一系列处理)，对一张图片而言，算法可能会输出M个预测框
            every element in predict has shape [M, 5], here number 5 represent [xim, ymin, xmax, ymax, conf]
        :param ground_truth: [batch_size, ]
            与predict一一对应的每张图片的ground truth bbox，GT_bbox的数目很可能与算法预测的不一致
            every element in ground_truth has shape [N, 4], here number 4 represent [xmin, ymin, xmax, ymax]
        :param iou_threshold: scalar
            对于elevenInterpolation，iou_threshold一般取0.5
            对于everyInterpolation，iou_threshold可以取任意[0, 1]之间的数
        """
        assert len(predict) == len(ground_truth)
        # [batch_size, M]
        self.pred = predict
        # [batch_size, N]
        self.gt_box = ground_truth
        self.iou_threshold = iou_threshold

        self.ap_dict = self.make_ap_dict()
        self.precision, self.recall = self.compute_pr(self.ap_dict)
        self.elevenPointAP = self.elevenPointInterpolation()
        self.everyPointAP = self.everyPointInterpolation()

    def make_ap_dict(self):
        ap_dict = defaultdict(list)
        for pred, gt_box in zip(self.pred, self.gt_box):
            pred, gt_box = np.asarray(pred), np.asarray(gt_box)
            tpfp, conf, gt_num = self.get_tpfp(pred[:, -1], pred[:, :-1], gt_box)
            ap_dict['tpfp'].extend(tpfp)
            ap_dict['conf'].extend(conf)
            ap_dict['gt_num'].append(gt_num)
        return ap_dict

    def get_tpfp(self, pred_conf, pred_box, gt_box):
        """
        每次调用只处理一张图片的预测结果，主要功能是判断该张图片中每个预测框为TP还是FP
        :param pred_conf: [M, 1]
        :param pred_box: [M, 4]
        :param gt_box: [N, 4]
        :return:
        """
        if len(pred_box) != 0:
            assert pred_conf.shape[0] == pred_box.shape[0]
            gt_num = gt_box.shape[0]
            # [M, 4] & [N, 4] -> [M, N]
            ious = self.iou(pred_box, gt_box)
            iou_thresd_mask = np.greater(ious, self.iou_threshold)
            # [M, N] & [M, 1] -> [M, N]
            max_iou_mask = np.equal(ious, np.expand_dims(np.max(ious, axis=-1), axis=1))
            pred2gt_mask = np.logical_and(iou_thresd_mask, max_iou_mask)
            tpfp_mask, descend_index = self.make_pr_mask(pred_conf, pred2gt_mask)
            tpfp = np.sum(tpfp_mask, axis=-1)
            conf = pred_conf[descend_index]
            return tpfp, conf, gt_num
        else:
            return 0., 0., gt_box.shape[0]

    @staticmethod
    def iou(pred_box, gt_box):
        """
        :param pred_box: [M, 4]
        :param gt_box: [N, 4]
        :return: [M, N]
        """
        # expand dim for broadcast computing
        # shape: [M, 1, 4]
        pred_box = np.expand_dims(pred_box, axis=1)
        # shape: [M, 1]
        pred_box_area = np.prod(pred_box[..., [2, 3]] - pred_box[..., [0, 1]] + 1, axis=-1)
        # shape: [N,]
        gt_box_area = np.prod(gt_box[:, [2, 3]] - gt_box[:, [0, 1]] + 1, axis=-1)
        # [M, 1] & [N,] -> [M, N]
        intersection_xmin = np.maximum(pred_box[..., 0], gt_box[:, 0])
        intersection_ymin = np.maximum(pred_box[..., 1], gt_box[:, 1])
        intersection_xmax = np.minimum(pred_box[..., 2], gt_box[:, 2])
        intersection_ymax = np.minimum(pred_box[..., 3], gt_box[:, 3])
        # [M, N] & [M, N] -> [M, N]
        intersection_w = np.maximum(0., intersection_xmax - intersection_xmin + 1)
        intersection_h = np.maximum(0., intersection_ymax - intersection_ymin + 1)
        intersection_area = intersection_w * intersection_h
        # [M, N] & [M, 1] & [N,] & [M, N] -> [M, N]
        ious = intersection_area / (pred_box_area + gt_box_area - intersection_area)
        return ious

    @staticmethod
    def make_pr_mask(pred_conf, pred2gt_mask):
        """
        每次调用只处理一张图片的预测结果，主要功能是确保每个预测框最多只负责一个gt_box的预测
        :param pred_conf:
        :param pred2gt_mask:
        :return:
        """
        descend_index = np.argsort(pred_conf)[::-1]
        pred2gt_mask = pred2gt_mask[descend_index]
        for i in range(pred2gt_mask.shape[0]):
            nonzero_index = pred2gt_mask[i].nonzero()[0]
            if nonzero_index.shape[0] != 0:
                assert nonzero_index.shape[0] == 1
                column_id = nonzero_index[0]
                pred2gt_mask[(i+1):, column_id] = False
        return pred2gt_mask, descend_index

    @staticmethod
    def compute_pr(ap_dict):
        """
        对得到的tpfp_list按照pred_conf降序排序后，分别计算每个位置的precision和recall
        :param ap_dict:
        :return:
        """
        sorted_order = np.argsort(ap_dict['conf'])[::-1]
        all_gt_num = np.sum(ap_dict['gt_num'])
        ordered_tpfp = np.array(ap_dict['tpfp'])[sorted_order]
        recall = np.cumsum(ordered_tpfp) / all_gt_num
        ones = np.ones_like(recall)
        precision = np.cumsum(ordered_tpfp) / np.cumsum(ones)
        return precision, recall

    def elevenPointInterpolation(self):
        precision_list = []
        interpolation_points = np.arange(0, 1.1, 0.1)
        for point in interpolation_points:
            index = np.greater(self.recall, point)
            if index.sum() > 0:
                precision_list.append(np.max(self.precision[self.recall >= point]))
            else:
                precision_list.append(0.)
        return np.mean(precision_list)

    def everyPointInterpolation(self):
        last_recall = 0.
        auc = 0.
        for recall in self.recall:
            precision = np.max(self.precision[self.recall >= recall])
            auc += (recall - last_recall) * precision
            last_recall = recall
        return auc


if __name__ == '__main__':
    # 7张图片的ground truth
    # box format: [xmin, ymin, w/2, h/2]
    gt = [[[25, 16, 38, 56], [129, 123, 41, 62]],
          [[123, 11, 43, 55], [38, 132, 59, 45]],
          [[16, 14, 35, 48], [123, 30, 49, 44], [99, 139, 47, 47]],
          [[53, 42, 40, 52], [154, 43, 31, 34]],
          [[59, 31, 44, 51], [48, 128, 34, 52]],
          [[36, 89, 52, 76], [62, 58, 44, 67]],
          [[28, 31, 55, 63], [58, 67, 50, 58]]]

    # 7张图片的预测结果
    pred = [[[5, 67, 31, 48, 0.88], [119, 111, 40, 67, 0.7], [124, 9, 49, 67, 0.8]],
            [[64, 111, 64, 58, 0.71], [26, 140, 60, 47, 0.54], [19, 18, 43, 35, 0.74]],
            [[109, 15, 77, 39, 0.18], [86, 63, 46, 45, 0.67], [160, 62, 36, 53, 0.38], [105, 131, 47, 47, 0.91], [18, 148, 40, 44, 0.44]],
            [[83, 28, 28, 26, 0.35], [28, 68, 42, 67, 0.78], [87, 89, 25, 39, 0.45], [10, 155, 60, 26, 0.14]],
            [[50, 38, 28, 46, 0.62], [95, 11, 53, 28, 0.44], [29, 131, 72, 29, 0.95], [29, 163, 72, 29, 0.23]],
            [[43, 48, 74, 38, 0.45], [17, 155, 29, 35, 0.84], [95, 110, 25, 42, 0.43]],
            [[16, 20, 101, 88, 0.48], [33, 116, 37, 49, 0.95]]]

    for i, arr in enumerate(gt):
        arr = np.array(arr)
        arr[:, [2, 3]] = arr[:, [0, 1]] + arr[:, [2, 3]]
        gt[i] = arr

    for i, arr in enumerate(pred):
        arr = np.array(arr)
        arr[:, [2, 3]] = arr[:, [0, 1]] + arr[:, [2, 3]]
        pred[i] = arr

    MAP = mAP(pred, gt, 0.3)
    print('Precision: ', np.around(MAP.precision, 2))
    print('Recall: ', np.around(MAP.recall, 2))
    print('AP: %.2f %%' % (MAP.elevenPointAP * 100))
    print('mAP: %.2f %%' % (MAP.everyPointAP * 100))


