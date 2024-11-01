from abc import ABC, abstractmethod
import torch
import torchaudio.transforms as T
from utils.utils_data import MinMaxScaler, MinMaxArgs


class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, device, seq_len, delay, embedding):
        super().__init__(device, seq_len)
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None

    def pad_to_square(self, x, mask=0):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0, max_side - rows, 0, max_side - cols)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal, pad=True, mask=0):

        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            # end = start + (self.embedding - 1) - missing_vals
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image

    def img_to_ts(self, img):
        img_non_square = self.unpad(img, self.img_shape)

        batch, channels, rows, cols = img_non_square.shape

        reconstructed_x_time_series = torch.zeros((batch, channels, self.seq_len))

        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed_x_time_series[:, :, start:end] = img_non_square[:, :, :, i]

        ### SPECIAL CASE
        start = (cols - 1) * self.delay
        end = reconstructed_x_time_series[:, :, start:].shape[-1]
        reconstructed_x_time_series[:, :, start:] = img_non_square[:, :, :end, cols - 1]
        reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)

        return reconstructed_x_time_series.cuda()


class STFTEmbedder(TsImgEmbedder):
    """
    STFT transformation
    """

    def __init__(self, device, seq_len, n_fft, hop_length):
        super().__init__(device, seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_real, self.max_real, self.min_imag, self.max_imag = None, None, None, None

    def cache_min_max_params(self, train_data):
        """
        Args:
            train_data: training timeseries dataset. shape: B*L*K
        this function initializes the min and max values for the real and imaginary parts.
        we'll use this function only once, before the training loop starts.
        """
        real, imag = self.stft_transform(train_data)
        # compute and cache min and max values
        real, min_real, max_real = MinMaxScaler(real.numpy(), True)
        imag, min_imag, max_imag = MinMaxScaler(imag.numpy(), True)
        self.min_real, self.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
        self.min_imag, self.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)

    def stft_transform(self, data):
        """
        Args:
            data: time series data. Shape: B*L*K
        Returns:
            real and imaginary parts of the STFT transformation
        """
        data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
        spec = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, center=True, power=None).to(data.device)
        transformed_data = spec(data)
        return transformed_data.real, transformed_data.imag

    def ts_to_img(self, signal):
        assert self.min_real is not None, "use init_norm_args() to compute scaling arguments"
        # convert to complex spectrogram
        real, imag = self.stft_transform(signal)
        # MinMax scaling
        real = (MinMaxArgs(real, self.min_real.to(self.device), self.max_real.to(self.device)) - 0.5) * 2
        imag = (MinMaxArgs(imag, self.min_imag.to(self.device), self.max_imag.to(self.device)) - 0.5) * 2
        # stack real and imag parts
        stft_out = torch.cat((real, imag), dim=1)
        return stft_out

    def img_to_ts(self, x_image):
        n_fft = self.n_fft
        hop_length, length = self.hop_length, self.seq_len
        min_real, max_real, min_imag, max_imag = self.min_real.to(
            self.device), self.max_real.to(
            self.device), \
            self.min_imag.to(self.device), self.max_imag.to(
            self.device)
        # -- combine real and imaginary parts --
        split = torch.split(x_image, x_image.shape[1] // 2,
                            dim=1)  # x_image.shape[1] is twice the size of the original dim

        real, imag = split[0], split[1]
        unnormalized_real = ((real / 2) + 0.5) * (max_real - min_real) + min_real
        unnormalized_imag = ((imag / 2) + 0.5) * (max_imag - min_imag) + min_imag
        unnormalized_stft = torch.complex(unnormalized_real, unnormalized_imag)
        # -- inverse stft --
        ispec = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length, center=True).to(self.device)

        x_time_series = ispec(unnormalized_stft, length)

        return torch.permute(x_time_series, (0, 2, 1))  # B*L*K(C)
