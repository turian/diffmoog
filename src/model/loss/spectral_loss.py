from typing import Callable, Union

import torch
import torchaudio
from torch import nn

from model.loss.spectral_loss_presets import loss_presets
from synth.synth_constants import SynthConstants

loss_type_to_function = {'mag': lambda x: x,
                         'delta_time': lambda x: torch.diff(x, n=1, dim=1),
                         'delta_freq': lambda x: torch.diff(x, n=1, dim=2),
                         'cumsum_time': lambda x: torch.cumsum(x, dim=2),
                         'cumsum_freq': lambda x: torch.cumsum(x, dim=1),
                         'logmag': lambda x: torch.log(x + 1)}


class SpectralLoss(nn.Module):
    """From DDSP code:
    https://github.com/magenta/ddsp/blob/8536a366c7834908f418a6721547268e8f2083cc/ddsp/losses.py#L144"""
    """Multiscale spectrogram loss.
    This loss is the bread-and-butter of comparing two audio signals. It offers
    a range of options to compare spectrograms, many of which are redunant, but
    emphasize different aspects of the signal. By far, the most common comparisons
    are magnitudes (mag_weight) and log magnitudes (logmag_weight).
    """

    def __init__(self, loss_type: str, loss_preset: Union[str, dict], synth_constants: SynthConstants, device='cuda:0'):
        """Constructor, set loss weights of various components.
    Args:
      fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
        spectrogram has a time-frequency resolution trade-off based on fft size,
        so comparing multiple scales allows multiple resolutions.
      loss_type: One of 'SPECTROGRAM', 'MEL_SPECTROGRAM', or 'BOTH'.
      mag_weight: Weight to compare linear magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to peak magnitudes than log
        magnitudes.
      delta_time_weight: Weight to compare the first finite difference of
        spectrograms in time. Emphasizes changes of magnitude in time, such as
        at transients.
      delta_freq_weight: Weight to compare the first finite difference of
        spectrograms in frequency. Emphasizes changes of magnitude in frequency,
        such as at the boundaries of a stack of harmonics.
      cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
        across frequency for each slice in time. Similar to a 1-D Wasserstein
        loss, this hopefully provides a non-vanishing gradient to push two
        non-overlapping sinusoids towards eachother.
      logmag_weight: Weight to compare log magnitudes of spectrograms. Core
        audio similarity loss. More sensitive to quiet magnitudes than linear
        magnitudes.
      loudness_weight: Weight to compare the overall perceptual loudness of two
        signals. Very high-level loss signal that is a subset of mag and
        logmag losses.
    """

        super().__init__()

        self.preset = loss_presets[loss_preset] if isinstance(loss_preset, str) else loss_preset
        self.device = device
        self.sample_rate = synth_constants.sample_rate

        if self.preset['multi_spectral_loss_type'] == 'L1':
            self.criterion = nn.L1Loss()
        elif self.preset['multi_spectral_loss_type'] == 'L2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("unknown loss type")

        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self.spectrogram_ops = {}
        for size in self.preset['fft_sizes']:
            if loss_type == 'BOTH' or loss_type == 'SPECTROGRAM':
                spec_transform = torchaudio.transforms.Spectrogram(n_fft=size, hop_length=int(size / 4), power=2.0).to(self.device)

                self.spectrogram_ops[f'{size}_spectrogram'] = spec_transform

            if loss_type == 'BOTH' or loss_type == 'MEL_SPECTROGRAM':
                mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=size,
                                                                          hop_length=int(size / 4), n_mels=256,
                                                                          power=2.0).to(self.device)

                self.spectrogram_ops[f'{size}_mel'] = mel_spec_transform


    def call(self, target_audio, predicted_audio, step: int, return_spectrogram: bool = False):
        """ execute multi-spectral loss computation between two audio signals

        Args:
          :param target_audio: target audio signal
          :param predicted_audio: predicted audio signal
          :param summary_writer: tensorboard summary writer
        """
        loss = 0.0

        # Compute loss for each fft size.
        loss_dict, weighted_loss_dict = {}, {}
        spectrograms_dict = {}
        for loss_name, loss_op in self.spectrogram_ops.items():

            n_fft = loss_op.n_fft

            target_mag = loss_op(target_audio.float())
            value_mag = loss_op(predicted_audio.float())

            c_loss = 0.0
            for loss_type, pre_loss_fn in loss_type_to_function.items():
                raw_loss, weighted_loss = self.calc_loss(loss_type, pre_loss_fn, target_mag, value_mag, n_fft, step)

                if weighted_loss == 0:
                    continue

                loss_dict[f"{loss_name}_{loss_type}"] = raw_loss
                weighted_loss_dict[f"{loss_name}_{loss_type}"] = weighted_loss

                c_loss += weighted_loss

            loss += c_loss

            spectrograms_dict[loss_name] = {'pred': value_mag.detach(), 'target': target_mag.detach()}

        if return_spectrogram:
            return loss, loss_dict, weighted_loss_dict, spectrograms_dict

        return loss, loss_dict, weighted_loss_dict

    def calc_loss(self, loss_type: str, pre_loss_fn: Callable, target_mag: torch.Tensor, value_mag: torch.Tensor,
                  n_fft: int, step: int) -> (float, float):

        # If loss weight is 0, or warmup for this loss type is not over
        if self.preset.get(f'multi_spectral_{loss_type}_weight', 0) == 0 or \
                step < self.preset.get(f'multi_spectral_{loss_type}_warmup', -1):
            return 0, 0

        # Prepare spectrograms according to loss type
        target = pre_loss_fn(target_mag)
        value = pre_loss_fn(value_mag)

        # Calculate raw loss value
        loss_val = self.criterion(target, value)

        # Weight loss according to preset
        weighted_loss_val = loss_val * self.preset[f'multi_spectral_{loss_type}_weight']

        # Normalize by n_fft if required
        if self.preset['normalize_loss_by_nfft']:
            n_fft_normalization_factor = (300.0 / n_fft)
            if loss_type in ['cumsum_time']:
                n_fft_normalization_factor = 1
            elif loss_type not in ['delta_time', 'logmag']:
                n_fft_normalization_factor = n_fft_normalization_factor ** 2
            weighted_loss_val *= n_fft_normalization_factor

        # Normalize to increase gradually through training if required
        if self.preset.get(f'multi_spectral_{loss_type}_gradual', False):
            warmup = self.preset.get(f'multi_spectral_{loss_type}_warmup_steps')
            gradual_loss_factor = min((step / warmup), 1)
            weighted_loss_val *= gradual_loss_factor

        return loss_val, weighted_loss_val

class ControlSpectralLoss(nn.Module):
    """
    This loss aims at comparing control signals such as LFOs. Control signals contain only low frequecny components, and
     therefore need to be downsampled and compared using a Spectrogram with lower nyquist frequency.
    """

    def __init__(self,
                 signal_duration: float,
                 loss_type: str,
                 preset_name: str,
                 synth_constants: SynthConstants,
                 device='cuda:0'):
        """Constructor, set loss weights of various components.
    Args:

    """

        super().__init__()

        self.preset = loss_presets[preset_name]
        self.device = device
        self.sample_rate = synth_constants.sample_rate

        if self.preset['multi_spectral_loss_type'] == 'L1':
            self.criterion = nn.L1Loss()
        elif self.preset['multi_spectral_loss_type'] == 'L2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("unknown loss type")

        target_sample_rate = synth_constants.lfo_signal_sampling_rate
        self.resample = torchaudio.transforms.Resample(self.sample_rate, target_sample_rate)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=128,
                                                             win_length=128,
                                                             hop_length=512,
                                                             center=True,
                                                             pad_mode="reflect",
                                                             power=2.0).to(self.device)
        self.db = torchaudio.transforms.AmplitudeToDB(stype='magnitude')

    def call(self, target_control_signal, predicted_control_signal, step: int, return_spectrogram: bool = False):
        """ execute spectral loss computation between two control signals

        Args:
          :param target_control_signal: target control signal
          :param predicted_control_signal: predicted control signal
          :param summary_writer: tensorboard summary writer
        """
        loss = 0.0

        resampled_target_control_signal = self.resample(target_control_signal)
        resampled_predicted_control_signal = self.resample(predicted_control_signal)
        target_mag = self.spectrogram(resampled_target_control_signal.float())
        value_mag = self.spectrogram(resampled_predicted_control_signal.float())

        # fig, axs = plt.subplots(1, 1, figsize=(13, 2.5))
        # im = axs.imshow(librosa.power_to_db(target_mag[0].detach().cpu().numpy()), origin="lower", aspect="auto",
        #                 cmap='inferno')

        c_loss = 0.0
        n_fft = 128
        loss_dict, weighted_loss_dict = {}, {}
        spectrograms_dict = {}
        for loss_type, pre_loss_fn in loss_type_to_function.items():
            raw_loss, weighted_loss = self.calc_loss(loss_type, pre_loss_fn, target_mag, value_mag, n_fft, step)

            if weighted_loss == 0:
                continue

            loss_dict[f"{loss_type}"] = raw_loss
            weighted_loss_dict[f"{loss_type}"] = weighted_loss

            c_loss += weighted_loss

        loss += c_loss

        spectrograms_dict['control_signal_spectrograms'] = {'pred': value_mag.detach(),
                                                            'target': target_mag.detach()}

        if return_spectrogram:
            return loss, loss_dict, weighted_loss_dict, spectrograms_dict

        return loss, loss_dict, weighted_loss_dict

    def calc_loss(self, loss_type: str, pre_loss_fn: Callable, target_mag: torch.Tensor, value_mag: torch.Tensor,
                  n_fft: int, step: int) -> (float, float):

        # If loss weight is 0, or warmup for this loss type is not over
        if self.preset.get(f'multi_spectral_{loss_type}_weight', 0) == 0 or \
                step < self.preset.get(f'multi_spectral_{loss_type}_warmup', -1):
            return 0, 0

        # Prepare spectrograms according to loss type
        target = pre_loss_fn(target_mag)
        value = pre_loss_fn(value_mag)

        # Calculate raw loss value
        loss_val = self.criterion(target, value)

        # Weight loss according to preset
        weighted_loss_val = loss_val * self.preset[f'multi_spectral_{loss_type}_weight']

        # Normalize by n_fft if required
        if self.preset['normalize_loss_by_nfft']:
            n_fft_normalization_factor = (300.0 / n_fft)
            if loss_type in ['cumsum_time']:
                n_fft_normalization_factor = 1
            elif loss_type not in ['delta_time', 'logmag']:
                n_fft_normalization_factor = n_fft_normalization_factor ** 2
            weighted_loss_val *= n_fft_normalization_factor

        # Normalize to increase gradually through training if required
        if self.preset.get(f'multi_spectral_{loss_type}_gradual', False):
            warmup = self.preset.get(f'multi_spectral_{loss_type}_warmup_steps')
            gradual_loss_factor = min((step / warmup), 1)
            weighted_loss_val *= gradual_loss_factor

        return loss_val, weighted_loss_val



