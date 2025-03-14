API_ROOT = "http://localhost:6500/api/v1/"
TIMEOUT = 10
OPTIMIZER_TYPE = ["Adam", "AdamW"]
OPTIMIZER_PARAMS = {
    "Adam": {
        "weight_decay":  (0.0, "%0.2f", 0.0, 1.0), # (default_value, accuracy, min_value, max_value)
    },
    "AdamW": {
        "weight_decay": (0.01, "%0.2f", 0.0, 1.0),
    },
}
LOSS_TYPE = ["bce", "focal"]
LOSS_PARAMS = {
    "bce": {
        "pos_weight": (None, "%0.2f", 0.0, 1.0),
    },
    "focal": {
        "alpha": (0.25, "%0.2f", 0.0, 1.0),
        "gamma": (2.0, "%0.2f", 0.0, 5.0),
    },
}
LR_SCHEDULER_TYPE = ["disable", "plateau"]
LR_SCHEDULER_PARAMS = {
    "disable": {

    },
    "plateau": {
        "factor": (0.1, "%0.1f", 0.0, 1.0),
        "patience": (5, "%d", 0, 10),
        "threshold": (1e-4, "%0.4f", 0.0, 1.0),
        "min_lr": (1e-6, "%0.6f", 0.0, 1.0),
        "eps": (1e-6, "%0.6f", 0.0, 1.0),
    },
}
