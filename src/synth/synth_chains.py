"""
This file contains the synth chains that are used to generate the synth patches.
One can add new synth chains here, and use them for dataset generation and experiments.
"""

BASIC_FLOW_LFO_DECOUPLING = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True},
    {'index': (1, 1), 'operation': 'fm', 'audio_input': [[1, 0]], 'output': [0, 2]},
    {'index': (1, 2), 'operation': None, 'audio_input': None},
    {'index': (2, 2), 'operation': None, 'audio_input': None},
    {'index': (0, 2), 'operation': 'mix', 'audio_input': [[0, 1], [1, 1]]},
    {'index': (0, 3), 'operation': 'filter', 'default_connection': True},
    {'index': (0, 4), 'operation': 'env_adsr', 'default_connection': True}
]

BASIC_FLOW = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True},
    {'index': (0, 1), 'operation': 'fm_sine', 'default_connection': True},
    {'index': (1, 0), 'operation': 'lfo', 'default_connection': True},
    {'index': (1, 1), 'operation': 'fm_square', 'audio_input': [[1, 0]], 'output': [0, 2]},
    {'index': (2, 0), 'operation': 'lfo', 'default_connection': True},
    {'index': (2, 1), 'operation': 'fm_saw', 'audio_input': [[2, 0]], 'output': [0, 2]},
    {'index': (1, 2), 'operation': None, 'audio_input': None},
    {'index': (2, 2), 'operation': None, 'audio_input': None},
    {'index': (0, 2), 'operation': 'mix', 'audio_input': [[0, 1], [1, 1], [2, 1]]},
    {'index': (0, 3), 'operation': 'env_adsr', 'default_connection': True},
    {'index': (0, 4), 'operation': 'lowpass_filter', 'default_connection': True}
]

MODULAR = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'audio_input': None, 'control_input': None, 'outputs': [(0, 6), (1, 1)], 'switch_outputs': True, 'active_prob': 0.75},
    {'index': (1, 1), 'operation': 'fm_lfo', 'audio_input': None, 'control_input': [[0, 0]], 'outputs': [(0, 2), (1, 2), (2, 2)], 'switch_outputs': True, 'allow_multiple': False, 'active_prob': 0.75},
    {'index': (0, 2), 'operation': 'fm_sine', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
    {'index': (1, 2), 'operation': 'fm_saw', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
    {'index': (2, 2), 'operation': 'fm_square', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 3), 'operation': 'mix', 'audio_input': [[0, 2], [1, 2], [2, 2]], 'control_input': None, 'outputs': [(0, 4)]},
    {'index': (0, 4), 'operation': 'env_adsr', 'audio_input': [[0, 3]], 'control_input': None, 'outputs': [(0, 5)]},
    {'index': (0, 5), 'operation': 'lowpass_filter', 'audio_input': [[0, 4]], 'control_input': None, 'outputs': [(0, 6)]},
    {'index': (0, 6), 'operation': 'tremolo', 'audio_input': [[0, 5]], 'control_input': [[0, 0]], 'outputs': None, 'active_prob': 0}
]

NO_FILTER = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'audio_input': None, 'control_input': None, 'outputs': [(0, 6), (1, 1)], 'switch_outputs': True, 'active_prob': 0.75},
    {'index': (1, 1), 'operation': 'fm_lfo', 'audio_input': None, 'control_input': [[0, 0]], 'outputs': [(0, 2), (1, 2), (2, 2)], 'switch_outputs': True, 'allow_multiple': False, 'active_prob': 0.75},
    {'index': (0, 2), 'operation': 'fm_sine', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
    {'index': (1, 2), 'operation': 'fm_saw', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
    {'index': (2, 2), 'operation': 'fm_square', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 3), 'operation': 'mix', 'audio_input': [[0, 2], [1, 2], [2, 2]], 'control_input': None, 'outputs': [(0, 4)]},
    {'index': (0, 4), 'operation': 'env_adsr', 'audio_input': [[0, 3]], 'control_input': None, 'outputs': [(0, 6)]},
    {'index': (0, 5), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 6), 'operation': 'tremolo', 'audio_input': [[0, 4]], 'control_input': [[0, 0]], 'outputs': None, 'active_prob': 0}
]

MODULAR_NEW = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'audio_input': None, 'control_input': None, 'outputs': [[0, 6], [1, 1]], 'switch_outputs': True, 'active_prob': 0.75},
    {'index': (1, 1), 'operation': 'fm_lfo', 'audio_input': None, 'control_input': [[0, 0]], 'outputs': [[0, 2], [1, 2], [2, 2]], 'switch_outputs': True, 'active_prob': 0.75},
    {'index': (0, 2), 'operation': 'fm_sine', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [[0, 3]]},
    {'index': (1, 2), 'operation': 'fm_saw', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [[0, 3]]},
    {'index': (2, 2), 'operation': 'fm_square', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [[0, 3]]},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 3), 'operation': 'mix', 'audio_input': [[0, 2], [1, 2], [2, 2]], 'control_input': None, 'outputs': [[0, 4]]},
    {'index': (0, 4), 'operation': 'env_adsr', 'audio_input': [[0, 3]], 'control_input': None, 'outputs': [[0, 5]]},
    {'index': (0, 5), 'operation': 'lowpass_filter_adsr', 'audio_input': [[0, 4]], 'control_input': None, 'outputs': [[0, 6]]},
    {'index': (0, 6), 'operation': 'tremolo', 'audio_input': [[0, 5]], 'control_input': [[0, 0]], 'outputs': None, 'active_prob': 0}
]

FM_DX7 = [
    {'index': (0, 0), 'operation': 'osc', 'audio_input': None, 'control_input': None, 'outputs': [[0, 1]]},
    {'index': (1, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 2), 'operation': 'fm_sine', 'audio_input': None, 'control_input': [[0, 1]], 'outputs': [[0, 3]]},
    {'index': (1, 2), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 2), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 1), 'operation': 'env_adsr', 'audio_input': [[0, 0]], 'control_input': None, 'outputs': [[0, 2]]},
    {'index': (1, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 3), 'operation': None, 'audio_input': [[0, 2]], 'control_input': None, 'outputs': [[0, 4]]},
    {'index': (0, 4), 'operation': None, 'audio_input': [[0, 3]], 'control_input': None, 'outputs': [[0, 5]]},
    {'index': (0, 5), 'operation': None, 'audio_input': [[0, 4]], 'control_input': None, 'outputs': [[0, 6]]},
    {'index': (0, 6), 'operation': None, 'audio_input': [[0, 5]], 'control_input': None, 'outputs': None}
]

REDUCED = [
    {'index': (1, 2), 'operation': 'fm_saw', 'audio_input': None, 'control_input': None, 'outputs': [[0, 3]]},
    {'index': (2, 2), 'operation': 'fm_square', 'audio_input': None, 'control_input': None, 'outputs': [[0, 3]]},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (2, 3), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 3), 'operation': 'mix', 'audio_input': [[1, 2], [2, 2]], 'control_input': None, 'outputs': [[0, 4]]},
    {'index': (0, 4), 'operation': 'env_adsr', 'audio_input': [[0, 3]], 'control_input': None, 'outputs': [[0, 5]]},
    {'index': (0, 5), 'operation': 'lowpass_filter_adsr', 'audio_input': [[0, 4]], 'control_input': None, 'outputs': [[0, 6]]},
]

REDUCED_SIMPLE_FILTER = [
    {'index': (0, 0), 'operation': 'osc_saw_no_activeness', 'audio_input': None, 'control_input': None, 'outputs': [[0, 1]]},
    {'index': (1, 0), 'operation': 'osc_square_no_activeness', 'audio_input': None, 'control_input': None, 'outputs': [[0, 1]]},
    {'index': (0, 1), 'operation': 'mix', 'audio_input': [[0, 0], [1, 0]], 'control_input': None, 'outputs': [[0, 2]]},
    {'index': (0, 2), 'operation': 'env_adsr', 'audio_input': [[0, 1]], 'control_input': None, 'outputs': [[0, 3]]},
    {'index': (0, 3), 'operation': 'lowpass_filter', 'audio_input': [[0, 2]], 'control_input': None, 'outputs': None},
]

REDUCED_SIMPLE_OSC = [
    {'index': (0, 0), 'operation': 'osc_saw_no_activeness_cont_freq', 'audio_input': None, 'control_input': None, 'outputs': [[0, 1]]},
    {'index': (1, 0), 'operation': 'osc_square_no_activeness_cont_freq', 'audio_input': None, 'control_input': None, 'outputs': [[0, 1]]},
    {'index': (0, 1), 'operation': 'mix', 'audio_input': [[0, 0], [1, 0]], 'control_input': None, 'outputs': [[0, 2]]},
    {'index': (0, 2), 'operation': 'env_adsr', 'audio_input': [[0, 1]], 'control_input': None, 'outputs': [[0, 3]]},
    {'index': (0, 3), 'operation': 'lowpass_filter', 'audio_input': [[0, 2]], 'control_input': None, 'outputs': None},
]

SAW_SQUARE_MIX = [
    {'index': (0, 0), 'operation': 'saw_square_osc', 'default_connection': True},
]

SAW_SQUARE_MIX_FILTER = [
    {'index': (0, 0), 'operation': 'saw_square_osc',  'audio_input': None, 'outputs': [[0, 1]]},
    {'index': (0, 1), 'operation': 'lowpass_filter', 'audio_input': [[0, 0]], 'outputs': [[0, 2]]},
]

BASIC_FLOW_NO_ADSR = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True},
    {'index': (1, 1), 'operation': 'fm', 'audio_input': [[1, 0]], 'output': [0, 2]},
    {'index': (1, 2), 'operation': None, 'audio_input': None},
    {'index': (0, 2), 'operation': 'mix', 'audio_input': [[0, 1], [1, 1]]},
    {'index': (0, 3), 'operation': 'filter', 'default_connection': True},
]

BASIC_FLOW_NO_FILTER = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True},
    {'index': (1, 1), 'operation': 'fm', 'audio_input': [[1, 0]], 'output': [0, 2]},
    {'index': (1, 2), 'operation': None, 'audio_input': None},
    {'index': (0, 2), 'operation': 'mix', 'audio_input': [[0, 1], [1, 1]]},
    {'index': (0, 3), 'operation': 'amplitude_shape', 'default_connection': True},
]

BASIC_FLOW_NO_ADSR_NO_FILTER = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True},
    {'index': (1, 1), 'operation': 'fm', 'audio_input': [[1, 0]], 'output': [0, 2]},
    {'index': (1, 2), 'operation': None, 'audio_input': None},
    {'index': (0, 2), 'operation': 'mix', 'audio_input': [[0, 1], [1, 1]]},
]

SINE_LFO = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True, 'active_prob': 1},
]

OSC = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True, 'active_prob': 1}
]

TWO_LFO_SAW = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'audio_input': None, 'control_input': None, 'outputs': [(1, 1)],
     'switch_outputs': True, 'active_prob': 0.25},
    {'index': (1, 1), 'operation': 'fm_lfo', 'audio_input': None, 'control_input': [[0, 0]],
     'outputs': [(0, 2)], 'switch_outputs': True, 'allow_multiple': False, 'active_prob': 0.5},
    {'index': (0, 2), 'operation': 'fm_saw', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
]

LFO_SAW = [
    {'index': (0, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 1), 'operation': 'lfo', 'audio_input': None, 'control_input': None,
     'outputs': [(0, 2)], 'switch_outputs': False, 'allow_multiple': False, 'active_prob': 0.5},
    {'index': (0, 2), 'operation': 'fm_saw', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
]

LFO_SQUARE = [
    {'index': (0, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 1), 'operation': 'lfo', 'audio_input': None, 'control_input': None,
     'outputs': [(0, 2)], 'switch_outputs': False, 'allow_multiple': False, 'active_prob': 0.5},
    {'index': (0, 2), 'operation': 'fm_square', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
]

SURROGATE_LFO_SAW = [
    {'index': (0, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 1), 'operation': 'surrogate_lfo', 'audio_input': None, 'control_input': None,
     'outputs': [(0, 2)], 'switch_outputs': False, 'allow_multiple': False, 'active_prob': 0.5},
    {'index': (0, 2), 'operation': 'surrogate_fm_saw', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
]

SURROGATE_LFO_SINE = [
    {'index': (0, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 1), 'operation': 'surrogate_lfo', 'audio_input': None, 'control_input': None,
     'outputs': [(0, 2)], 'switch_outputs': False, 'allow_multiple': False, 'active_prob': 1},
    {'index': (0, 2), 'operation': 'surrogate_fm_sine', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
]

LFO_SIN = [
    {'index': (0, 0), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (0, 1), 'operation': None, 'audio_input': None, 'control_input': None, 'outputs': None},
    {'index': (1, 1), 'operation': 'lfo', 'audio_input': None, 'control_input': None,
     'outputs': [(0, 2)], 'switch_outputs': False, 'allow_multiple': False, 'active_prob': 1},
    {'index': (0, 2), 'operation': 'fm_sine', 'audio_input': None, 'control_input': [[1, 1]], 'outputs': [(0, 3)]},
]

OSC_FILTER = [
    {'index': (0, 0), 'operation': 'osc', 'audio_input': None, 'control_input': None, 'outputs': [(0, 1)]},
    {'index': (0, 1), 'operation': 'lowpass_filter', 'audio_input': [[0, 0]], 'control_input': None, 'outputs': [(0, 2)]},
]

OSC_ADSR = [
    {'index': (0, 0), 'operation': 'osc', 'audio_input': None, 'control_input': None, 'outputs': [(0, 1)]},
    {'index': (0, 1), 'operation': 'env_adsr', 'audio_input': [[0, 0]], 'control_input': None, 'outputs': [(0, 2)]},
]

OSC_TREMOLO = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'audio_input': None, 'control_input': None, 'outputs': [(0, 2)], 'switch_outputs': False, 'active_prob': 1},
    {'index': (0, 1), 'operation': 'osc', 'audio_input': None, 'control_input': None, 'outputs': [(0, 2)]},
    {'index': (0, 2), 'operation': 'tremolo', 'audio_input': [[0, 1]], 'control_input': [[0, 0]], 'outputs': [(0, 3)], 'active_prob': 1}
]

NON_SINE_LFO = [
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'default_connection': True},
]

DOUBLE_FM_ONLY = [
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True},
    {'index': (1, 1), 'operation': 'fm', 'audio_input': [], 'output': [0, 2]},
    {'index': (1, 2), 'operation': None, 'audio_input': None},
    {'index': (0, 2), 'operation': 'mix', 'audio_input': [[0, 1], [1, 1]]},
]

LFO = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True}
]

OSC = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True}
]

FM = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True}
]

FM_ONLY = [
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True}
]

FM_FILTER = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True},
    {'index': (0, 2), 'operation': 'filter', 'default_connection': True}
]

FILTER_ONLY = [
    {'index': (0, 2), 'operation': 'filter', 'default_connection': True}
]


FM_FILTER_ADSR = [
    {'index': (0, 0), 'operation': 'lfo', 'default_connection': True},
    {'index': (0, 1), 'operation': 'fm', 'default_connection': True},
    {'index': (0, 2), 'operation': 'filter', 'default_connection': True},
    {'index': (0, 3), 'operation': 'env_adsr', 'default_connection': True}
]

DOUBLE_LFO = [
    {'index': (0, 0), 'operation': 'lfo_sine', 'default_connection': True},
    {'index': (1, 0), 'operation': 'lfo_non_sine', 'output': [0, 1]},
    {'index': (0, 1), 'operation': 'mix', 'audio_input': [[0, 0], [1, 0]], 'output': [0, 2]},
    {'index': (1, 1), 'operation': None, 'audio_input': None}
]

OSC_AMPLITUDE_SHAPER = [
    {'index': (0, 0), 'operation': 'osc', 'default_connection': True},
    {'index': (0, 1), 'operation': 'env_adsr', 'default_connection': True}
]

OSC_FILTER_SHAPER = [
    {'index': (0, 0), 'operation': 'osc', 'audio_input': None, 'control_input': None, 'outputs': [(0, 1)]},
    {'index': (0, 1), 'operation': 'lowpass_filter_adsr', 'audio_input': [(0, 0)], 'control_input': None, 'outputs': [(0, 2)]}
]

synth_chains_dict = {'BASIC_FLOW': BASIC_FLOW,
                      'LFO': LFO,
                      'OSC': OSC,
                      'FM': FM,
                      'BASIC_FLOW_LFO_DECOUPLING': BASIC_FLOW_LFO_DECOUPLING,
                      'BASIC_FLOW_NO_ADSR': BASIC_FLOW_NO_ADSR,
                      'BASIC_FLOW_NO_ADSR_NO_FILTER': BASIC_FLOW_NO_ADSR_NO_FILTER,
                      'FM_FILTER_ADSR': FM_FILTER_ADSR,
                      'FM_FILTER': FM_FILTER,
                      'DOUBLE_LFO': DOUBLE_LFO,
                      'FM_ONLY': FM_ONLY,
                      'FILTER_ONLY': FILTER_ONLY,
                      'OSC_AMPLITUDE_SHAPER': OSC_AMPLITUDE_SHAPER,
                      'OSC_FILTER_SHAPER': OSC_FILTER_SHAPER,
                      'BASIC_FLOW_NO_FILTER': BASIC_FLOW_NO_FILTER,
                      'DOUBLE_FM_ONLY': DOUBLE_FM_ONLY,
                      'NON_SINE_LFO': NON_SINE_LFO,
                      'SINE_LFO': SINE_LFO,
                      'MODULAR': MODULAR,
                      'MODULAR_NEW': MODULAR_NEW,
                      'REDUCED': REDUCED,
                      'NO_FILTER': NO_FILTER,
                      'TWO_LFO_SAW': TWO_LFO_SAW,
                      'LFO_SAW': LFO_SAW,
                      'LFO_SQUARE': LFO_SQUARE,
                      'SAW_SQUARE_MIX': SAW_SQUARE_MIX,
                      'LFO_SIN': LFO_SIN,
                      'SURROGATE_LFO_SAW': SURROGATE_LFO_SAW,
                      'SURROGATE_LFO_SINE': SURROGATE_LFO_SINE,
                      'SAW_SQUARE_MIX_FILTER': SAW_SQUARE_MIX_FILTER,
                      'OSC_FILTER': OSC_FILTER,
                      'OSC_ADSR': OSC_ADSR,
                      'OSC_TREMOLO': OSC_TREMOLO,
                      'FM_DX7': FM_DX7,
                      'REDUCED_SIMPLE_FILTER': REDUCED_SIMPLE_FILTER,
                      'REDUCED_SIMPLE_OSC': REDUCED_SIMPLE_OSC
                      }
