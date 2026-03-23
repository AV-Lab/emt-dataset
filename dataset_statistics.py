#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset statistics script:
- prints tracking / prediction / intention statistics
- saves only 2 plots:
    1) Intention Distribution
    2) Number of Agents
"""

import os
import json
import argparse
import statistics as s
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


VEHICLES = {
    "Car",
    "Large_vehicle",
    "Medium_vehicle",
    "Bus",
    "Emergency_vehicle",
    "Small_motorised_vehicle",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_intention_plot(intention_counter, save_path):
    labels = [k for k, _ in intention_counter.most_common()]
    counts = [intention_counter[k] for k in labels]

    df = pd.DataFrame({
        "Intention": labels,
        "Count": counts,
    })

    sns.set_theme(style="whitegrid", context="paper")

    plt.figure(figsize=(12.5, 6.8))
    ax = sns.barplot(
        data=df,
        x="Intention",
        y="Count",
        color=sns.color_palette()[0]
    )

    ax.set_yscale("log")
    ax.set_title("Intention Distribution", fontsize=21)
    ax.set_xlabel("Intention Class", fontsize=20)
    ax.set_ylabel("Count (log scale)", fontsize=20)

    plt.xticks(rotation=40, ha="right", fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def save_agents_per_scene_plot(frame_scene_counts, save_path):
    counter = Counter(frame_scene_counts)
    labels = sorted(counter.keys())
    counts = [counter[k] for k in labels]
    total = sum(counts)
    percents = [c / total for c in counts]

    df = pd.DataFrame({
        "Number of Agents": [str(x) for x in labels],
        "Percent of Scenes": percents,
    })

    sns.set_theme(style="whitegrid", context="paper")

    plt.figure(figsize=(15, 7.8))
    ax = sns.barplot(
        data=df,
        x="Number of Agents",
        y="Percent of Scenes",
        color=sns.color_palette()[0]
    )

    ax.set_title("Distribution of Agents", fontsize=25)
    ax.set_xlabel("Number of Agents", fontsize=23, labelpad=25)
    ax.set_ylabel("Percent of Scenes", fontsize=23)

    plt.xticks(rotation=90, fontsize=22)
    plt.yticks(fontsize=22)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


################################## Tracking statistics ##################################

def tracking_statistics(annot_dir):
    annotations = os.listdir(annot_dir)

    vehciles_tracklets = 0
    pedestrian_tracklets = 0
    cyclist_tracklets = 0
    motorbike_tracklets = 0
    tracklets_length = []

    for ann in annotations:
        ann_file = os.path.join(annot_dir, ann)
        tracklets = {}

        with open(ann_file, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                frame_id = parts[0]
                track_id = parts[1]
                obj_class = parts[2]

                if track_id not in tracklets:
                    tracklets[track_id] = {"obj_class": obj_class, "frames": []}
                tracklets[track_id]["frames"].append(frame_id)

        for _, v in tracklets.items():
            if v["obj_class"] in VEHICLES:
                vehciles_tracklets += 1
            elif v["obj_class"] == "Pedestrian":
                pedestrian_tracklets += 1
            elif v["obj_class"] == "Cyclist":
                cyclist_tracklets += 1
            elif v["obj_class"] == "Motorbike":
                motorbike_tracklets += 1

            tracklets_length.append(len(v["frames"]))

    print("Tracking Benchmark Statistics:")
    print(f"Total number of vehciles: {vehciles_tracklets}")
    print(f"Total number of pedestrians: {pedestrian_tracklets}")
    print(f"Total number of motorbikes: {motorbike_tracklets}")
    print(f"Total number of cyclists: {cyclist_tracklets}")
    print(f"Mean tracklet length: {s.mean(tracklets_length)} \n")


################################## Prediction statistics ##################################

def prediction_statistics(annot_dir, figures_dir):
    annotations = os.listdir(annot_dir)

    vehciles_pred = 0
    pedestrian_pred = 0
    cyclist_pred = 0
    motorbike_pred = 0
    total = 0

    prediction_lengths = []
    frame_scene_counts = []

    for ann in annotations:
        ann_file = os.path.join(annot_dir, ann)

        with open(ann_file, "r") as file:
            data = json.load(file)

        frame_counter = defaultdict(int)

        for _, v in data.items():
            cls = v["class"]
            frames = v["frames"]

            if cls == "Pedestrian":
                pedestrian_pred += 1
            elif cls == "Cyclist":
                cyclist_pred += 1
            elif cls == "Motorbike":
                motorbike_pred += 1
            elif cls in VEHICLES:
                vehciles_pred += 1

            prediction_lengths.append(len(frames))
            total += 1

            for fr in frames:
                frame_counter[fr] += 1

        frame_scene_counts.extend(frame_counter.values())

    print("Prediction Benchmark Statistics:")
    print(f"Total number of agents: {total}")
    print(f"Total number of vehciles: {vehciles_pred}")
    print(f"Total number of pedestrians: {pedestrian_pred}")
    print(f"Total number of motorbikes: {motorbike_pred}")
    print(f"Total number of cyclists: {cyclist_pred}")
    print(f"Mean prediction sequence length: {s.mean(prediction_lengths)} \n")

    save_agents_per_scene_plot(
        frame_scene_counts,
        os.path.join(figures_dir, "agents_distribution.png"),
    )


################################## Intention statistics ##################################

def intention_statistics(annot_dir, figures_dir):
    annotations = os.listdir(annot_dir)

    total = 0
    intention_counter = Counter()

    for ann in annotations:
        ann_file = os.path.join(annot_dir, ann)

        with open(ann_file, "r") as file:
            data = json.load(file)

        total += len(data.keys())

        for _, v in data.items():
            intentions = v.get("intention", [])
            for inten in intentions:
                intention_counter[inten] += 1

    print("Intention Benchmark Statistics:")
    print(f"Total number of sequences: {total}")
    print("Intention distribution:")
    for k, v in intention_counter.most_common():
        print(f"{k}: {v}")
    print()

    save_intention_plot(
        intention_counter,
        os.path.join(figures_dir, "intention_distribution.png"),
    )


################################## Main ##################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset statistics and save figures")
    parser.add_argument("--dataset_dir", type=str, required=True, help="dataset root path")
    args = parser.parse_args()

    figures_dir = os.path.join(args.dataset_dir, "statistics_figures")
    ensure_dir(figures_dir)

    tracking_annot_dir = os.path.join(args.dataset_dir, "annotations", "tracking", "kitti")
    prediction_annot_dir = os.path.join(args.dataset_dir, "annotations", "prediction")
    intention_annot_dir = os.path.join(args.dataset_dir, "annotations", "intention")

    tracking_statistics(tracking_annot_dir)
    prediction_statistics(prediction_annot_dir, figures_dir)
    intention_statistics(intention_annot_dir, figures_dir)

    print(f"Saved figures to: {figures_dir}")