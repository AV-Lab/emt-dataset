#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:41:33 2025

@author: nadya
"""


CLASS_LABELS = {
    "reversing": 0,
    "turn-right": 1,
    "turn-left": 2,
    "merge-right": 3,
    "merge-left": 4,
    "braking": 5,
    "stopped": 6,
    "lane-keeping": 7,
    "walking": 8,
    "crossing": 9,
    "waiting_to_cross": 10
}


LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}  