#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loss functions for Few-shot Learning
"""

import torch
import torch.nn as nn


def get_entropy(p_softmax):
    """
    Calculate entropy for entropy minimization
    """
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))


def get_causality_loss(x_IN_entropy, x_useful_entropy, x_useless_entropy):
    """
    Calculate causality loss for domain adaptation
    """
    ranking_loss = torch.nn.SoftMarginLoss()
    y = torch.ones_like(x_IN_entropy)
    return ranking_loss(x_IN_entropy - x_useful_entropy, y) + ranking_loss(x_useless_entropy - x_IN_entropy, y)
