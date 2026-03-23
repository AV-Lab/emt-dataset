#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 18:29:23 2026

@author: nadya
"""


import re
from typing import Dict, Tuple


def parse_split_metadata(path: str):
    train, test = {}, {} #dict[str, tuple[int, int]]
    current = None
    pattern = re.compile(r"^(?P<video>\S+)\s*\((?P<w>\d+)\s*,\s*(?P<h>\d+)\)$")

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.lower().startswith("train"):
                current = "train"
                continue

            if line.lower().startswith("test"):
                current = "test"
                continue

            match = pattern.match(line)
            if match is None:
                raise ValueError(f"Invalid metadata line: {line}")

            video_name = match.group("video")
            width = int(match.group("w"))
            height = int(match.group("h"))

            if current == "train":
                train[video_name] = (width, height)
            elif current == "test":
                test[video_name] = (width, height)

    return train, test