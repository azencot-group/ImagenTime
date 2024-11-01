import numpy as np
import torchaudio.transforms as transforms
import os
import sys
import torch
import torch.utils.data as Data

from data.data_provider.data_factory import data_provider
from data.long_range import parse_datasets

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world data.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    assert data_name in ['stock', 'energy', 'metro']

    if data_name == 'stock':
        ori_data = np.loadtxt('./data/short_range/stock_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('./data/short_range/energy_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'metro':
        ori_data = np.loadtxt('./data/short_range/metro_data.csv', delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the data
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the data (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


def gen_dataloader(args):
    if args.dataset == 'sine':
        args.dataset_size = 10000
        ori_data = sine_data_generation(args.dataset_size, args.seq_len, args.input_channels)
        ori_data = torch.Tensor(np.array(ori_data))
        train_set = Data.TensorDataset(ori_data)

    elif args.dataset in ['stock', 'energy']:
        ori_data = real_data_loading(args.dataset, args.seq_len)
        ori_data = torch.Tensor(np.array(ori_data))
        train_set = Data.TensorDataset(ori_data)

    elif args.dataset in ['mujoco']:
        train_set = MujocoDataset(args.seq_len, args.dataset, args.path, 0.0)

    elif args.dataset in ['solar_weekly', 'fred_md', 'nn5_daily', 'temperature_rain', 'traffic_hourly', 'kdd_cup']:
        ori_data = parse_datasets(args.dataset, args.batch_size, args.device, args)
        ori_data = torch.stack(ori_data)
        args.seq_len = ori_data.shape[1]  # update seq_len to match the dataset
        full_len = ori_data.shape[0]
        randperm = torch.randperm(full_len)
        train_data = ori_data[randperm[:int(full_len * 0.8)]]
        test_data = ori_data[randperm[int(full_len * 0.8):]]
        train_set = Data.TensorDataset(train_data)
        test_set = Data.TensorDataset(test_data)
        train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers)
        test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        return train_loader, test_loader

    elif args.dataset in ['physionet', 'climate']:
        train_loader, test_loader = parse_datasets(args.dataset, args.batch_size, args.device, args)
        return train_loader, test_loader

    elif args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        train_data, train_loader = data_provider(args, flag='train')
        test_data, test_loader = data_provider(args, flag='test')
        return train_loader, test_loader

    train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    # for the short-term time series benchmark, the entire dataset for both training and testing
    return train_loader, train_loader


def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def stft_transform(data, args):
    data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
    n_fft = args.n_fft
    hop_length = args.hop_length
    spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True, power=None)
    transformed_data = spec(data)
    real, min_real, max_real = MinMaxScaler(transformed_data.real.numpy(), True)
    real = (real - 0.5) * 2
    imag, min_imag, max_imag = MinMaxScaler(transformed_data.imag.numpy(), True)
    imag = (imag - 0.5) * 2
    # saving min and max values, we will need them for inverse transform
    args.min_real, args.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
    args.min_imag, args.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)
    return torch.Tensor(real), torch.tensor(imag)


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


class MujocoDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, path, missing_rate=0.0):
        # import pdb;pdb.set_trace()
        import pathlib
        here = pathlib.Path(__file__).resolve().parent.parent
        base_loc = here / 'data'
        loc = pathlib.Path(path)
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)
        else:
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)
            if not os.path.exists(loc):
                os.mkdir(loc)
            loc = here / 'data' / data_name
            tensors = load_data(loc)
            time = tensors['train_X'][:, :, :1].cpu().numpy()
            data = tensors['train_X'][:, :, 1:].reshape(-1, 14).cpu().numpy()

            self.original_sample = []
            norm_data = normalize(data)
            norm_data = norm_data.reshape(4620, seq_len, 14)
            idx = torch.randperm(len(norm_data))

            for i in range(len(norm_data)):
                self.original_sample.append(norm_data[idx[i]].copy())
            self.X_mean = np.mean(np.array(self.original_sample), axis=0).reshape(1,
                                                                                  np.array(self.original_sample).shape[
                                                                                      1],
                                                                                  np.array(self.original_sample).shape[
                                                                                      2])
            generator = torch.Generator().manual_seed(56789)
            for i in range(len(norm_data)):
                removed_points = torch.randperm(norm_data[i].shape[0], generator=generator)[
                                 :int(norm_data[i].shape[0] * missing_rate)].sort().values
                norm_data[i][removed_points] = float('nan')
            norm_data = np.concatenate((norm_data, time), axis=2)
            self.samples = []
            for i in range(len(norm_data)):
                self.samples.append(norm_data[idx[i]])

            self.samples = np.array(self.samples)

            norm_data_tensor = torch.Tensor(self.samples[:, :, :-1]).float().cuda()

            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.samples[:, :, -1][:, -1]).float()
            self.original_sample = torch.tensor(self.original_sample)
            self.samples = torch.tensor(self.samples)
            loc = here / 'data' / (data_name + str(missing_rate))
            save_data(loc, data=self.samples,
                      original_data=self.original_sample
                      )
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

    def __getitem__(self, index):
        return self.original_sample[index], self.samples[index]

    def __len__(self):
        return len(self.samples)
