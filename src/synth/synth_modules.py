#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:41:38 2021

@author: Moshe Laufer, Noy Uzrad
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Sequence

import numpy as np

import torch
import math

from torchaudio.functional.filtering import lowpass_biquad, highpass_biquad

from model import helper
import julius
from julius.lowpass import lowpass_filter_new
from model.gumble_softmax import gumbel_softmax
from synth.synth_constants import SynthConstants
from utils.types import TensorLike

try:
    from functorch import vmap
    has_vmap = True
except ModuleNotFoundError:
    has_vmap = False


PI = math.pi
TWO_PI = 2 * PI


class SynthModule(ABC):

    def __init__(self, name: str, device: str, synth_structure: SynthConstants):
        self.synth_structure = synth_structure
        self.device = device
        self.name = name

    @abstractmethod
    def process_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                      sample_rate: int, signal_duration: float, batch_size: int = 1) -> torch.Tensor:
        pass

    @staticmethod
    def _mix_waveforms(waves_tensor: torch.Tensor, raw_waveform_selector: Union[str, Sequence[str], TensorLike],
                       type_indices: Dict[str, int]) -> torch.Tensor:

        if isinstance(raw_waveform_selector, str):
            idx = type_indices[raw_waveform_selector]
            return waves_tensor[idx]

        if isinstance(raw_waveform_selector[0], str):
            oscillator_tensor = torch.stack([waves_tensor[type_indices[wt]][i] for i, wt in
                                             enumerate(raw_waveform_selector)])
            return oscillator_tensor

        oscillator_tensor = 0
        for i in range(len(waves_tensor)):
            oscillator_tensor += raw_waveform_selector.transpose()[i].unsqueeze(1) * waves_tensor[i]

        return oscillator_tensor

    def _standardize_input(self, input_val, requested_dtype, requested_dims: int, batch_size: int,
                           value_range: Tuple = None) -> torch.Tensor:

        # Single scalar input value
        if isinstance(input_val, (float, np.floating, int, np.int)):
            assert batch_size == 1, f"Input expected to be of batch size {batch_size} but is scalar"
            input_val = torch.tensor(input_val, dtype=requested_dtype, device=self.device)

        # List, ndarray or tensor input
        if isinstance(input_val, (np.ndarray, list)):
            output_tensor = torch.tensor(input_val, dtype=requested_dtype, device=self.device)
        elif torch.is_tensor(input_val):
            output_tensor = input_val.to(dtype=requested_dtype, device=self.device)
        else:
            raise TypeError(f"Unsupported input of type {type(input_val)} to synth module")

        # Add batch dim if doesn't exist
        if output_tensor.ndim == 0:
            output_tensor = torch.unsqueeze(output_tensor, dim=0)
        if output_tensor.ndim == 1 and len(output_tensor) == batch_size:
            output_tensor = torch.unsqueeze(output_tensor, dim=1)
        elif output_tensor.ndim == 1 and len(output_tensor) != batch_size:
            assert batch_size == 1, f"Input expected to be of batch size {batch_size} but is of batch size 1, " \
                                    f"shape {output_tensor.shape}"
            output_tensor = torch.unsqueeze(output_tensor, dim=0)

        assert output_tensor.ndim == requested_dims, f"Input has unexpected number of dimensions: {output_tensor.ndim}"

        if value_range is not None:
            assert torch.all(output_tensor >= value_range[0]) and torch.all(output_tensor <= value_range[1]), \
                f"Parameter value outside of expected range {value_range}"

        return output_tensor

    def _verify_input_params(self, params_to_test: dict):

        expected_params = self.synth_structure.modular_synth_params[self.name]
        assert set(params_to_test.keys()) == set(expected_params),\
            f'Expected input parameters {expected_params} but got {list(params_to_test.keys())}'

    def _process_active_signal(self, active_vector: TensorLike, batch_size: int) -> torch.Tensor:

        if active_vector is None:
            ret_active_vector = torch.ones([batch_size, 1], dtype=torch.long, device=self.device)
            return ret_active_vector

        if not isinstance(active_vector, torch.Tensor):
            active_vector = torch.tensor(active_vector)

        if active_vector.dtype == torch.bool:
            ret_active_vector = active_vector.long().unsqueeze(1).to(self.device)
            return ret_active_vector

        active_vector_gumble = gumbel_softmax(active_vector, hard=True, device=self.device)
        ret_active_vector = active_vector_gumble[:, :1]

        return ret_active_vector


class Oscillator(SynthModule):

    def __init__(self, name: str, device: str, synth_structure: SynthConstants, waveform: str = None):

        super().__init__(name=name, device=device, synth_structure=synth_structure)

        self.waveform = waveform
        self.wave_type_indices = {k: torch.tensor(v, dtype=torch.long, device=self.device).squeeze()
                                  for k, v in self.synth_structure.wave_type_dict.items()}

    def process_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                      sample_rate: int, signal_duration: float, batch_size: int = 1) -> torch.Tensor:
        """Creates a basic oscillator.

            Retrieves a waveform shape and attributes, and construct the respected signal

            Args:
                self: Self object
                amp: batch of n amplitudes in range [0, 1]
                freq: batch of n frequencies in range [0, 22000]
                phase: batch of n phases in range [0, 2pi], default is 0
                waveform: batch of n strings, one of ['sine', 'square', 'triangle', 'sawtooth'] or a probability vector

            Returns:
                A torch tensor with the constructed signal

            Raises:
                ValueError: Provided variables are out of range
                :rtype: object
            """

        self._verify_input_params(params)

        active_signal = self._process_active_signal(params.get('active', None), batch_size)

        amp = self._standardize_input(params['amp'], requested_dtype=torch.float32, requested_dims=2,
                                      batch_size=batch_size)
        freq = self._standardize_input(params['freq'], requested_dtype=torch.float32, requested_dims=2,
                                       batch_size=batch_size)

        amp *= active_signal
        freq *= active_signal

        t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), requires_grad=True)
        wave_tensors = self._generate_wave_tensors(t, amp, freq, phase_mod=0)

        if self.waveform is not None:
            return wave_tensors[self.waveform]

        waves_tensor = torch.stack([wave_tensors['sine'], wave_tensors['square'], wave_tensors['saw']])
        oscillator_tensor = self._mix_waveforms(waves_tensor, params['waveform'], self.wave_type_indices)

        return oscillator_tensor

    def _generate_wave_tensors(self, t, amp, freq, phase_mod):

        wave_tensors = {}

        if self.waveform == 'sine' or self.waveform is None:
            sine_wave = amp * torch.sin(TWO_PI * freq * t + phase_mod)
            wave_tensors['sine'] = sine_wave

        if self.waveform == 'square' or self.waveform is None:
            square_wave = amp * torch.sign(torch.sin(TWO_PI * freq * t + phase_mod))
            wave_tensors['square'] = square_wave

        if self.waveform == 'saw' or self.waveform is None:
            sawtooth_wave = 2 * (t * freq - torch.floor(0.5 + t * freq))  # Sawtooth closed form

            # Phase shift (by normalization to range [0,1] and modulo operation)
            sawtooth_wave = (((sawtooth_wave + 1) / 2) + phase_mod / TWO_PI) % 1
            sawtooth_wave = amp * (sawtooth_wave * 2 - 1)  # re-normalization to range [-amp, amp]

            wave_tensors['saw'] = sawtooth_wave

        return wave_tensors


class FMOscillator(Oscillator):

    def __init__(self, name: str, device: str, synth_structure: SynthConstants, waveform: str = None):

        if waveform is not None:
            assert waveform in ['sine', 'square', 'saw'], f'Unexpected waveform {waveform} given to FMOscillator'

        super().__init__(name=name, device=device, synth_structure=synth_structure)

        self.waveform = waveform

    def process_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                      sample_rate: int, signal_duration: float, batch_size: int = 1) -> torch.Tensor:

        """
        Batch oscillator with FM modulation

        Creates an oscillator and modulates its frequency by a given modulator
        Args come as vector of values, with length of the numer of sounds to create
        params:
            self: Self object
            amp_c: Vector of amplitude in range [0, 1]
            freq_c: Vector of Frequencies in range [0, 22000]
            waveform: Vector of waveform with type from [sine, square, triangle, sawtooth]
            mod_index: Vector of modulation indexes, which affects the amount of modulation
            modulator: Vector of modulator signals, to affect carrier frequency

        Returns:
            A torch with the constructed FM signal

        Raises:
            ValueError: Provided variables are out of range
        """

        self._verify_input_params(params)

        active_signal = self._process_active_signal(params.get('active', None), batch_size)
        fm_active_signal = self._process_active_signal(params.get('fm_active', None), batch_size)

        parsed_params = {}
        for k in ['amp_c', 'freq_c', 'mod_index']:
            parsed_params[k] = self._standardize_input(params[k], requested_dtype=torch.float32, requested_dims=2,
                                                       batch_size=batch_size)

        parsed_params['freq_c'] = parsed_params['freq_c'] * active_signal
        parsed_params['mod_index'] = parsed_params['mod_index'] * fm_active_signal
        modulator_signal = modulator_signal * fm_active_signal

        t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), requires_grad=True)
        wave_tensors = self._generate_modulated_wave_tensors(t, modulator=modulator_signal, **parsed_params)

        if self.waveform is not None:
            return wave_tensors[self.waveform]

        waves_tensor = torch.stack([wave_tensors['sine'], wave_tensors['square'], wave_tensors['saw']])
        oscillator_tensor = self._mix_waveforms(waves_tensor, params['waveform'], self.wave_type_indices)

        return oscillator_tensor

    def _generate_modulated_wave_tensors(self, t, amp_c, freq_c, mod_index, modulator):

        wave_tensors = {}

        if self.waveform == 'sine' or self.waveform is None:
            fm_sine_wave = amp_c * torch.sin(TWO_PI * freq_c * t + mod_index * torch.cumsum(modulator, dim=1))
            wave_tensors['sine'] = fm_sine_wave

        if self.waveform == 'square' or self.waveform is None:
            fm_square_wave = amp_c * torch.sign(torch.sin(TWO_PI * freq_c * t + mod_index *
                                                          torch.cumsum(modulator, dim=1)))
            wave_tensors['square'] = fm_square_wave

        if self.waveform == 'saw' or self.waveform is None:
            fm_sawtooth_wave = 2 * (t * freq_c - torch.floor(0.5 + t * freq_c))
            fm_sawtooth_wave = (((fm_sawtooth_wave + 1) / 2) + mod_index * torch.cumsum(modulator, dim=1) / TWO_PI) % 1
            fm_sawtooth_wave = amp_c * (fm_sawtooth_wave * 2 - 1)
            wave_tensors['saw'] = fm_sawtooth_wave

        return wave_tensors


class ADSR(SynthModule):

    def __init__(self, device: str, synth_structure: SynthConstants):
        super().__init__(name='env_adsr', device=device, synth_structure=synth_structure)

    def process_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                      sample_rate: int, signal_duration: float, batch_size: int = 1) -> torch.Tensor:

        self._verify_input_params(params)

        parsed_params, num_timesteps = {}, {}
        total_time = 0
        for k in ['attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level']:
            parsed_params[k] = self._standardize_input(params[k], requested_dtype=torch.float64, requested_dims=2,
                                                       batch_size=batch_size)
            if k != 'sustain_level':
                num_timesteps[k] = torch.floor(sample_rate * signal_duration * parsed_params[k])
                total_time += parsed_params[k]

        if torch.any(total_time > signal_duration):
            raise ValueError("Provided ADSR durations exceeds signal duration")

        attack = torch.cat([torch.linspace(0, 1, int(attack_steps.item()), device=self.device) for attack_steps
                            in num_timesteps['attack_t']])
        decay = torch.cat([torch.linspace(1, sustain_val.int(), int(decay_steps), device=self.device) for
                           sustain_val, decay_steps in zip(parsed_params['sustain_level'], num_timesteps['decay_t'])])
        sustain = torch.cat([torch.full(sustain_steps, sustain_val, device=self.device) for sustain_steps, sustain_val
                             in zip(num_timesteps['sustain_t'], parsed_params['sustain_level'])])
        release = torch.cat([torch.linspace(sustain_val, 0, int(release_steps), device=self.device) for
                             sustain_val, release_steps in zip(parsed_params['sustain_level'],
                                                               num_timesteps['release_t'])])

        envelope = torch.cat((attack, decay, sustain, release))

        envelope_len = envelope.shape[0]
        signal_len = int(sample_rate * signal_duration)
        if envelope_len <= signal_len:
            padding = torch.zeros((signal_len - envelope_len), device=helper.get_device())
            envelope = torch.cat((envelope, padding))
        else:
            raise ValueError("Envelope length exceeds signal duration")

        enveloped_signal = input_signal * envelope

        return enveloped_signal


class Filter(SynthModule):

    def __init__(self, device: str, synth_structure: SynthConstants, filter_type: str = None):
        super().__init__(name='filter', device=device, synth_structure=synth_structure)

        if filter_type is not None:
            assert filter_type in ['lowpass', 'highpass'], f'Got unexpected filter type {filter_type} in Filter'
            self.name = f'{filter_type}_filter'

        self.filter_type = filter_type

        self.filter_type_indices = {k: torch.tensor(v, dtype=torch.long, device=self.device).squeeze()
                                    for k, v in synth_structure.filter_type_dict.items()}

    def process_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                      sample_rate: int, signal_duration: float, batch_size: int = 1) -> torch.Tensor:

        self._verify_input_params(params)

        filter_freq = self._standardize_input(params['filter_freq'], requested_dtype=torch.float64, requested_dims=2,
                                              batch_size=batch_size)

        assert torch.all(filter_freq <= (sample_rate / 2)), "Filter cutoff frequency higher then Nyquist." \
                                                            " Please check config"

        filtered_signal = self._generate_filtered_signal(input_signal, filter_freq, sample_rate, batch_size)
        for k, v in filtered_signal.items():
            if torch.any(torch.isnan(v)):
                raise RuntimeError("Synth filter module: Signal has NaN. Exiting...")

        if self.filter_type is not None:
            return filtered_signal[self.filter_type]

        waves_tensor = torch.stack([filtered_signal['lowpass'], filtered_signal['highpass']])
        filtered_signal_tensor = self._mix_waveforms(waves_tensor, params['filter_type'], self.filter_type_indices)

        return filtered_signal_tensor

    def _generate_filtered_signal(self, input_signal: torch.Tensor, filter_freq: torch.tensor, sample_rate: int,
                                  batch_size: int) -> dict:

        filtered_signals = {}
        if self.filter_type == 'lowpass' or self.filter_type is None:
            if has_vmap:
                low_pass_signals = vmap(lowpass_biquad)(input_signal.double(), cutoff_freq=filter_freq,
                                                        sample_rate=sample_rate)
            else:
                low_pass_signals = [self.low_pass(input_signal[i], filter_freq[i].cpu(), sample_rate) for i in
                                    range(batch_size)]
            filtered_signals['lowpass'] = torch.stack(low_pass_signals)

        if self.filter_type == 'highpass' or self.filter_type is None:
            if has_vmap:
                high_pass_signals = vmap(highpass_biquad)(input_signal.double(), cutoff_freq=filter_freq,
                                                          sample_rate=sample_rate)
            else:
                high_pass_signals = [self.high_pass(input_signal[i], filter_freq[i].cpu(), sample_rate) for i in
                                     range(batch_size)]
            filtered_signals['highpass'] = torch.stack(high_pass_signals)

        return filtered_signals

    @staticmethod
    def low_pass(input_signal, cutoff_freq, sample_rate, q=0.707, index=0):
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = lowpass_filter_new(input_signal, cutoff_freq / sample_rate)
            return filtered_waveform_new

    @staticmethod
    def high_pass(input_signal, cutoff_freq, sample_rate, q=0.707, index=0):
        if cutoff_freq == 0:
            return input_signal
        else:
            filtered_waveform_new = julius.highpass_filter_new(input_signal, cutoff_freq / sample_rate)
            return filtered_waveform_new


class Tremolo(SynthModule):

    def __init__(self, device: str, synth_structure: SynthConstants):
        super().__init__(name='tremolo', device=device, synth_structure=synth_structure)

    def process_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                      sample_rate: int, signal_duration: float, batch_size: int = 1) -> torch.Tensor:
        """tremolo effect for an input signal

            This is a kind of AM modulation, where the signal is multiplied as a whole by a given modulator.
            The modulator is shifted such that it resides in range [start, 1], where start is <1 - amount>.
            so start is > 0, such that the original amplitude of the input audio is preserved and there is no phase
            shift due to multiplication by negative number.

            params:
                self: Self object
                input_signal: Input signal to be used as carrier
                modulator_signal: modulator signal to modulate the input
                amount: amount of effect, in range [0, 1]


            Returns:
                A torch with the constructed AM signal

            Raises:
                ValueError: Provided variables are inappropriate
                ValueError: Amount is out of range [-1, 1]
            """

        self._verify_input_params(params)

        active_signal = self._process_active_signal(params.get('active', None), batch_size)

        amount = self._standardize_input(params['amount'], requested_dtype=torch.float64, requested_dims=2,
                                         batch_size=batch_size, value_range=(0, 1))

        modulator_signal = modulator_signal * active_signal
        tremolo = torch.add(torch.mul(amount, (modulator_signal + 1) / 2), (1 - amount))

        tremolo_signal = input_signal * tremolo

        return tremolo_signal


class Mix(SynthModule):

    def __init__(self, device: str, synth_structure: SynthConstants):
        super().__init__(name='mix', device=device, synth_structure=synth_structure)

    def process_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                      sample_rate: int, signal_duration: float, batch_size: int = 1) -> torch.Tensor:

        assert input_signal.shape[1] == batch_size and input_signal.shape[2] == sample_rate * signal_duration

        summed_signal = torch.sum(input_signal, dim=0)
        mixed_signal = summed_signal / torch.shape[0]

        return mixed_signal


class Noop(SynthModule):

    def process_sound(self, input_signal: torch.Tensor, modulator_signal: torch.Tensor, params: dict,
                      sample_rate: int, signal_duration: float, batch_size: int = 1) -> torch.Tensor:

        return input_signal


def get_synth_module(op_name: str, device: str, synth_structure: SynthConstants):

    op_name = op_name.lower()

    if op_name in ['osc', 'lfo', 'lfo_non_sine']:
        return Oscillator(op_name, device, synth_structure)
    elif op_name in ['lfo_sine', 'lfo_square', 'lfo_saw']:
        waveform = op_name.split('_')[1]
        return Oscillator(op_name, device, synth_structure, waveform)
    elif op_name in ['fm', 'fm_lfo']:
        return FMOscillator(op_name, device, synth_structure)
    elif op_name in ['fm_sine', 'fm_square', 'fm_saw']:
        waveform = op_name.split('_')[1]
        return FMOscillator(op_name, device, synth_structure, waveform)
    elif op_name in ['env_adsr']:
        return ADSR(device, synth_structure)
    elif op_name in ['filter', 'lowpass_filter', 'highpass_filter']:
        if op_name != 'filter':
            filter_type = op_name.split('_')[0]
            return Filter(device, synth_structure, filter_type)
        return Filter(device, synth_structure)
    elif op_name in ['tremolo']:
        return Tremolo(device, synth_structure)
    elif op_name in ['mix']:
        return Mix(device, synth_structure)
    elif op_name.lower() == 'none' or op_name is None:
        return Noop('none', device, synth_structure)
    else:
        raise ValueError(f"Unsupported synth module {op_name}")
