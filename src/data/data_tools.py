from __future__ import annotations

import random
import shutil
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union


import numpy as np
import tensorflow as tf
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from scipy.io import arff

Tensor = torch.Tensor



def get_eeg(data_dir: Path="../data/raw") -> Path:
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"  # noqa: E501
    datasetpath = tf.keras.utils.get_file(
        "eeg_data", origin=dataset_url, untar=False, cache_dir=data_dir
    )

    datasetpath = Path(datasetpath)
    logger.info(f"Data is downloaded to {datasetpath}.")
    return datasetpath





class BaseDataset:
    def __init__(self, datasetpath: Path) -> None:
        self.path = datasetpath
        self.data =  self.process_data()

    def process_data(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]


class EEGDataset(BaseDataset):

    def process_data(self) -> None:
        data = arff.loadarff(self.path)
        current_label = int(data[0][0][14]) #index 14 = label
        EEG_batch = [] #Lege lijst waarin meerdere observaties worden opgeslagen
        EEG_batches = [] #Lege lijst waarin meerdere batches in worden samengevoegd.
        for record in data[0]:
            if int(record[14]) == current_label:
                EEG_values = [] #Lege lijst waarin de EEG_values van een bepaalde observatie in kunnen worden opgeslagen.
                for index, i in enumerate(record):
                    if index != 14:
                        EEG_values.append(i)
                EEG_values = torch.Tensor(EEG_values)
                EEG_batch.append(EEG_values)
            else:
                EEG_batch_label = (current_label, torch.stack(EEG_batch ))
                EEG_batches.append(EEG_batch_label)
                current_label = int(record[14])
                EEG_batch = [] #Lege lijst waarin meerdere observaties in kunnen worden opgeslagen.
                EEG_values = [] #Lege lijst waarin de EEG_values van een bepaalde observatie in kunnen worden opgeslagen.
                for index, i in enumerate(record):
                    if index != 14:
                        EEG_values.append(i)
                EEG_values = torch.Tensor(EEG_values)
        EEG_batch_label = (current_label, torch.stack(EEG_batch))
        EEG_batches.append(EEG_batch_label)
        return EEG_batches






class BaseDataIteratorWindowing:

    def __init__(self, dataset: BaseDataset, window_size: int) -> None:
        self.dataset = dataset
        self.window_size = window_size
        self.data = self.window()   
   
    def __len__(self) -> int:
        return len(self.data)

    def window(self) -> None:
        dataset = self.dataset
        window_size = self.window_size
        window_full = []
        for i in range(len(dataset)):
            n_window = len(dataset[i][1]) - window_size + 1
            time = torch.arange(0, window_size).reshape(1, -1)
            window = torch.arange(0, n_window).reshape(-1, 1)
            idx = time + window
            window_out = dataset[i][1][idx]
            window_out_label = (dataset[i][0], window_out)
            window_full.append(window_out_label)
        return window_full
       
    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]





class BaseDataIteratorPaddingWindowing:

    def __init__(self, dataset: BaseDataset, window_size: int) -> None:
        self.data = dataset
        self.window_size = window_size
        self.dataset = self.padding()
        self.data = self.window()   
   
    def __len__(self) -> int:
        return len(self.data)

    def padding(self) -> None:
        data = self.data
        window_size = self.window_size
        list_padded = []
        for i in range(len(data)):
            len_chunck = len(data[i][1])
            padding_value = window_size - len_chunck
            if padding_value > 0:
                chuck_padding = F.pad(input=data[i][1], pad=(0, 0, 0, padding_value), mode='constant', value=0)
                padding_label = (data[i][0], chuck_padding)
                list_padded.append(padding_label)
            else:
                chunck = (data[i][0], data[i][1])
                list_padded.append(chunck)
        return list_padded


    def window(self) -> None:
        dataset = self.dataset
        window_size = self.window_size
        window_full = []
        for i in range(len(dataset)):
            n_window = len(dataset[i][1]) - window_size + 1
            time = torch.arange(0, window_size).reshape(1, -1)
            window = torch.arange(0, n_window).reshape(-1, 1)
            idx = time + window
            window_out = dataset[i][1][idx]
            window_out_label = (dataset[i][0], window_out)
            window_full.append(window_out_label)
        return window_full
       
    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]     