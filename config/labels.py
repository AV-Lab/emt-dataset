#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:41:33 2025

@author: nadya
"""


INTENTION_LABELS = {
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


AGENT_LABELS = {
    'Pedestrian': 1,
    'Cyclist': 2,
    'Motorbike': 3,
    'Small_motorised_vehicle': 4,
    'Medium_vehicle': 5,
    'Large_vehicle': 6,
    'Car': 7,
    'Bus': 8,
    'Emergency_vehicle': 9,
    'Vehicle_traffic_light': 10,
    'Other_traffic_light': 11
}
