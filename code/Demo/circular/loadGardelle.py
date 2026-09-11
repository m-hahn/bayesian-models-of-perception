import torch
import warnings
from util import MakeZeros

from util import MakeFloatTensor

from util import MakeLongTensor

from util import toFactor

STIMULUS_SPACE_MIN = 0.0
STIMULUS_SPACE_MAX = 360.0
SPAN_WARNING_TOLERANCE = 0.01 * (STIMULUS_SPACE_MAX - STIMULUS_SPACE_MIN)


def _range(values, name):
    if values.numel() == 0:
        raise ValueError(f"{name} is empty.")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return float(values.min()), float(values.max())


def _check_circular_bounds(values, name):
    minimum, maximum = _range(values, name)
    in_bounds = torch.logical_and(values >= STIMULUS_SPACE_MIN, values < STIMULUS_SPACE_MAX)
    if not bool(in_bounds.all()):
        bad_indices = torch.nonzero(~in_bounds, as_tuple=False).view(-1)[:5].cpu().numpy().tolist()
        raise ValueError(
            f"{name} must be in [{STIMULUS_SPACE_MIN}, {STIMULUS_SPACE_MAX}); "
            f"observed range [{minimum}, {maximum}], first bad indices {bad_indices}."
        )
    return minimum, maximum


def _warn_if_not_spanning_stimulus_space(values, name):
    minimum, maximum = _range(values, name)
    if minimum > STIMULUS_SPACE_MIN + SPAN_WARNING_TOLERANCE or maximum < STIMULUS_SPACE_MAX - SPAN_WARNING_TOLERANCE:
        warnings.warn(
            f"{name} spans [{minimum:.6g}, {maximum:.6g}], which does not cover the "
            f"declared circular stimulus space [{STIMULUS_SPACE_MIN}, {STIMULUS_SPACE_MAX}).",
            stacklevel=2,
        )


with open("data/GARDELLE/data.txt", "r") as inFile:
    data = [x.split("\t") for x in inFile.read().strip().split("\n")]
    header = [x.strip('"') for x in data[0]]
    header = dict(list(zip(header, range(len(header)))))
    print(header)
    data = data[1:]

# The original de Gardelle columns are axial orientations modulo 180 degrees.
# The circular model uses directional coordinates on [0, 360), so this loader
# doubles orientations before fitting.
sample = 2*MakeFloatTensor([float(x[header["rot"]]) for x in data])
responses = 2*MakeFloatTensor([float(x[header["resp_rot"]]) for x in data])
duration = MakeLongTensor(toFactor([int(x[header["targdur"]]) for x in data]))
Subject = MakeFloatTensor(toFactor([x[header["sub"]] for x in data]))

mask = torch.ByteTensor([float(x[header["rot"]]) not in [135, 90, 45, 0] for x in data])
print(mask.float().sum(), (1-mask.float()).sum())

sample = sample[mask]
responses = responses[mask]
duration = duration[mask]
Subject = Subject[mask]

_check_circular_bounds(sample, "Gardelle stimulus")
_check_circular_bounds(responses, "Gardelle response")
_warn_if_not_spanning_stimulus_space(sample, "Gardelle stimulus")


observations_x = sample
observations_y = responses
