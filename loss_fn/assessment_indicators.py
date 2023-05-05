import numpy as np
import torch
import torch.nn


def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def recall_fn(TP, TN, FP, FN):
    """正确预测正样本全度的能力，增加将正样本预测为正样本，即正样本被预测为正样本占总的正样本的比例。值越大，性能performance越好"""
    return TP / (TP + FP)


def precision_fn(TP, TN, FP, FN):
    """正确预测正样本精度的能力，即预测的正样本中有多少是真实的正样本。值越大，性能performance越好"""
    return TP / (TP + FP)


def FPR_fn(TP, TN, FP, FN):
    """ false alarm """
    """正确预测正样本纯度的能力，减少将负样本预测为正样本，即负样本被预测为正样本占总的负样本的比例。值越小，性能performance越好"""
    return FP / (FP + TN)


def FNR_fn(TP, TN, FP, FN):
    """ miss rate"""
    """正确预测负样本纯度的能力，减少将正样本预测为负样本，即正样本被预测为负样本占总的正样本的比例。值越小，性能performance越好"""
    return FN / (TP + FN)


def TNR_fn(TP, TN, FP, FN):
    """ specificity """
    """正确预测负样本全度的能力，增加将负样本预测为负样本，即负样本被预测为负样本占总的负样本的比例。值越大，性能performance越好   """
    return TN / (FP + TN)


def fx_score(TP, TN, FP, FN, x=1):
    """值越大，性能performance越好。F值可以平衡precision少预测为正样本和recall基本都预测为正样本的单维度指标缺陷。"""
    precision = precision_fn(TP, TN, FP, FN)
    recall = recall_fn(TP, TN, FP, FN)
    return (x**2+1)*(precision*recall)/(x**2*(precision+recall))






