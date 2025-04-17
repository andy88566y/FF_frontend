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
else:  # FF_ENV == Env.DEV:
    API_ROOT = f"http://localhost:{8580 + int(os.environ.get('DEV_NUM', '0'))}/api/v1/"


BLANK_MODEL = "[UNUSED]"

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
