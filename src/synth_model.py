from torch import nn
import torch
import helper
from torchsummary import summary
from synth import REGRESSION_PARAM_LIST, CLASSIFICATION_PARAM_LIST, WAVE_TYPE_DIC, FILTER_TYPE_DIC, OSC_FREQ_LIST

# todo: this is value from Valerio Tutorial. has to check
# LINEAR_IN_CHANNELS = 128 * 5 * 4
LINEAR_IN_CHANNELS = 4480
HIDDEN_IN_CHANNELS = 1000


class SynthNetwork(nn.Module):
    """
    CNN model to extract synth parameters from a batch of signals represented by mel-spectrograms

    :return: output_dic with classification and regression parameters.
                Concerning classification parameters:
                    'osc1_wave', 'osc2_wave' and 'filter freq' are returned as probability vectors.
                    size = (batch_size, #of_possibilities)

                    * Note:
                        In the synth module, in order to prevent if/else statements, the probability vectors will be
                        used to weight all possible wave and filter types in a superposition.
                        This operation is differential, while if/else statements are not


                    'osc1_freq' and 'osc2_freq' are returned as single values, by dot product.
                    size = (batch_size, 1)

                    The dot product is between learnt probabilities and fixed possible frequencies vector.
                    the learnt probabilities weight the possible frequencies.
                    * Note:
                        This technique allows the usage of a predicted frequency from a closed set of possible values in
                        the synth module, while preserving gradients computation.
                        Using argmax to predict a single value is not diffrentiable.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.classification_params = nn.ModuleDict([
            ['osc1_freq', nn.Linear(LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
            ['osc1_wave', nn.Linear(LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
            ['osc2_freq', nn.Linear(LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
            ['osc2_wave', nn.Linear(LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
            ['filter_type', nn.Linear(LINEAR_IN_CHANNELS, len(FILTER_TYPE_DIC))],
        ])
        self.regression_params = nn.Sequential(
            nn.Linear(LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
            nn.Linear(HIDDEN_IN_CHANNELS, len(REGRESSION_PARAM_LIST))
            )
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        if torch.any(torch.isnan(x)):
            a = 0

        # Apply different heads to predict each synth parameter
        output_dic = {}
        for out_name, lin in self.classification_params.items():
            # -----> do not use softmax if using CrossEntropyLoss()
            probabilities = self.softmax(lin(x))
            if out_name == 'osc1_freq' or out_name == 'osc2_freq':
                osc_freq_tensor = torch.tensor(OSC_FREQ_LIST, requires_grad=False, device=helper.get_device())
                output_dic[out_name] = torch.matmul(probabilities, osc_freq_tensor)
            else:
                output_dic[out_name] = probabilities

        x = self.regression_params(x)
        x = self.sigmoid(x)

        for index, param in enumerate(REGRESSION_PARAM_LIST):
            output_dic[param] = x[:, index]

        return output_dic


if __name__ == "__main__":
    synth_net = SynthNetwork()
    summary(synth_net.cuda(), (1, 64, 44))


