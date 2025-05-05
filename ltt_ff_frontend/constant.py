import os
from enum import Enum

from dotenv import load_dotenv
from loguru import logger


TIMEOUT = 10


### Working Environment
class Env(Enum):
    DEV = "dev"
    PROD = "prod"


# Load ENV variables from .env files
load_dotenv()

ff_env_value = os.environ.get("FF_ENV", Env.DEV.value)
try:
    FF_ENV = Env(ff_env_value)
except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning(f"Fail to extract the FF_ENV: {ff_env_value}. Error: {str(e)}. Force set the FF_ENV to {Env.DEV}.")
    FF_ENV = Env.DEV


if FF_ENV == Env.PROD:
    API_ROOT = "http://localhost:6500/api/v1/"
    RESTRICT_OUTPUT_DIR = True
else:  # FF_ENV == Env.DEV:
    API_ROOT = f"http://localhost:{8580 + int(os.environ.get('DEV_NUM', '0'))}/api/v1/"
    RESTRICT_OUTPUT_DIR = False


BLANK_MODEL = "[UNUSED]"

INFERENCE_DEFAULT_RESULT_DIR = "/mnt/dbpc/xxx"

OPTIMIZER_TYPE = ["Adam", "AdamW"]
OPTIMIZER_PARAMS = {
    "Adam": {
        "weight_decay": (0.0, "%0.3f", 0.0, 1.0),  # (default_value, accuracy, min_value, max_value)
    },
    "AdamW": {
        "weight_decay": (0.01, "%0.3f", 0.0, 1.0),
    },
}
LOSS_TYPE = ["bce", "focal"]
LOSS_PARAMS = {
    "bce": {
        "pos_weight": (None, "%d", 0, 100),
    },
    "focal": {
        "alpha": (0.25, "%0.2f", 0.0, 1.0),
        "gamma": (2.0, "%0.2f", 0.0, 10.0),
    },
}
LR_SCHEDULER_TYPE = ["disable", "plateau"]
LR_SCHEDULER_PARAMS = {
    "disable": {},
    "plateau": {
        "factor": (0.1, "%0.2f", 0.0, 1.0),
        "patience": (5, "%d", 0, 1000),
        "threshold": (1e-4, "%0.6f", 0.0, 1.0),
        "min_lr": (1e-6, "%0.8f", 0.0, 1.0),
        "eps": (1e-6, "%0.8f", 0.0, 1.0),
    },
}


class Site(Enum):
    F20 = "F20"
    F12 = "F12"
    F18A = "F18A"
    F18B = "F18B"
    F18EBO = "F18EBO"
    F15EBO = "F15EBO"


class Tool(Enum):
    X9U = "X9U"
    X912 = "X912"
    X9UHI = "X9UHI"
    X8U = "X8U"


class TechLayer(Enum):
    N2 = "N2"
    N3 = "N3"
    N4 = "N4"
    N5 = "N5"
    N28 = "N28"


class LayerGroup(Enum):
    OD = ["OD"]
    PO = ["PO"]
    ODPO = ["ODPO", "OD", "PO"]
    M0 = ["M0"]
    M2 = ["M2"]
    ME = ["ME"]
    M0M2 = ["M0M2", "M0", "M2", "ME"]
    CMD = ["CMD"]
    CMG = ["CMG"]
    CME = ["CME"]
    CUT = ["CUT", "CMD", "CMG", "CME"]
    VIA = ["VIA"]
    M1 = ["M1"]
    M3 = ["M3"]
    M4 = ["M4"]

    @staticmethod
    def get_base_layer_groups(main_layers: list["LayerGroup"]) -> list[str]:
        processed_layers, base_layers = [], []
        for layer in main_layers:
            if layer in processed_layers:
                pass

            processed_layers.append(layer)

            for sub_layer in layer.value:
                base_layers.append(sub_layer)

        return base_layers

    @staticmethod
    def get_full_layers() -> list[str]:
        """
        Get all layers that belong to ODPO, M0M2, and CUT
        """
        return LayerGroup.get_base_layer_groups(main_layers=[LayerGroup.ODPO, LayerGroup.M0M2, LayerGroup.CUT])

    @staticmethod
    def get_all_layers() -> list[str]:
        return LayerGroup.get_base_layer_groups(list(LayerGroup))


class PixelSize(Enum):
    NM40 = "40"
    NM45 = "45"
    NM50 = "50"
    NIL = None


class ModelType(Enum):
    BASE = "BASE"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


class MaskType(Enum):
    FLUSH = "flush"
    PRODUCTION = "production"
    TEST = "test"
