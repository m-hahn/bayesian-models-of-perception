import torch
import glob
import warnings
from util import MakeZeros

from util import MakeFloatTensor

from util import MakeLongTensor

from util import ToDevice

STIMULUS_SPACE_MIN = 0.0
STIMULUS_SPACE_MAX = 3.0
SPAN_WARNING_TOLERANCE = 0.01 * (STIMULUS_SPACE_MAX - STIMULUS_SPACE_MIN)


def _range(values, name):
    if values.numel() == 0:
        raise ValueError(f"{name} is empty.")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return float(values.min()), float(values.max())


def _check_interval_bounds(values, name, *, error=True):
    minimum, maximum = _range(values, name)
    in_bounds = torch.logical_and(values >= STIMULUS_SPACE_MIN, values <= STIMULUS_SPACE_MAX)
    if not bool(in_bounds.all()):
        bad_indices = torch.nonzero(~in_bounds, as_tuple=False).view(-1)[:5].cpu().numpy().tolist()
        message = (
            f"{name} must be in [{STIMULUS_SPACE_MIN}, {STIMULUS_SPACE_MAX}]; "
            f"observed range [{minimum}, {maximum}], first bad indices {bad_indices}."
        )
        if error:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    return minimum, maximum


def _warn_if_not_spanning_stimulus_space(values, name):
    minimum, maximum = _range(values, name)
    if minimum > STIMULUS_SPACE_MIN + SPAN_WARNING_TOLERANCE or maximum < STIMULUS_SPACE_MAX - SPAN_WARNING_TOLERANCE:
        warnings.warn(
            f"{name} spans [{minimum:.6g}, {maximum:.6g}], which does not cover the "
            f"declared interval stimulus space [{STIMULUS_SPACE_MIN}, {STIMULUS_SPACE_MAX}].",
            stacklevel=2,
        )


files = sorted(glob.glob("data/REMINGTON/Datafiles/RSG_*_100.mat"))
if not files:
    raise FileNotFoundError("No Remington data files found at data/REMINGTON/Datafiles/RSG_*_100.mat")

from scipy.io import loadmat

target = []
response = []

for f in files:
 annots = loadmat(f)
 sample = MakeFloatTensor(annots["sample"])
 responses = MakeFloatTensor(annots["response"])
 gain = MakeFloatTensor(annots["gain"])
 correct = MakeFloatTensor(annots["correct"])
 target.append(sample)
 response.append(responses)
 assert (correct-sample).abs().max() < .1, (correct-sample).abs().max()
 assert (gain-1).abs().max() < 0.1
target = torch.stack(target, dim=0)
response = torch.stack(response, dim=0)
subject = MakeFloatTensor(list(range(target.size()[0]))).view(-1, 1).expand(-1, target.size()[1])
target = target.view(-1).contiguous()
response = response.view(-1).contiguous()
Subject = subject.contiguous().view(-1).contiguous()

_check_interval_bounds(target, "Remington stimulus")
_check_interval_bounds(response, "Remington response", error=False)
_warn_if_not_spanning_stimulus_space(target, "Remington stimulus")

observations_x = target
observations_y = response

sample=target
