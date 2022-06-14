from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import gin
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.data import data_tools
from src.data.data_tools import PaddedDatagenerator, TSDataset

Tensor = torch.Tensor




def get_eeg(data_dir: Path="../../data/raw") -> Path:
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"  # noqa: E501
    datapath = tf.keras.utils.get_file(
        "eeg_data", origin=dataset_url, untar=True, cache_dir=data_dir
    )

    datapath = Path(datapath)
    logger.info(f"Data is downloaded to {datapath}.")
    return datapath
