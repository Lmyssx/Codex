from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class GateOperator:
    fc_hz: float
    bandwidth_hz: float
    dt_s: float
    nfft: int
    gate_bins: np.ndarray
    f_full_hz: np.ndarray
    mask: np.ndarray
    Q: np.ndarray
    singular_values: np.ndarray

@dataclass
class ObservationRecord:
    H_obs: np.ndarray
    f_obs_hz: np.ndarray
    operator: GateOperator
    W_frames: int
    stride_frames: int
    range_bins: int
    source_file: str
    posture: str
    observation_indices: np.ndarray
    sha256: str
    path: Path
    metadata: dict = field(default_factory=dict)

@dataclass
class Calibration:
    f_full_hz: np.ndarray
    system_response_full: np.ndarray
    residual_offset_full: np.ndarray
    metadata: dict

@dataclass
class MaterialModel:
    parameters: dict
    source: str
    units_raw: dict

@dataclass
class PosteriorContext:
    operator: GateOperator
    calibration: Calibration
    baseline: dict
    observed: np.ndarray
    covariance: np.ndarray
    bounds: np.ndarray
    material_sds: tuple = (0.1, 0.2)
    alpha0: float = 3.0
    beta0: float = 2.0
    g_air_m: float = 0.0
    calibration_delay_sd_s: float = 0.0
    long_operator: object = None
    long_baseline: object = None
    long_system: object = None
    long_offset: object = None
