API_ROOT = "http://localhost:6500/api/v1/"
TIMEOUT = 10
OPTIMIZER_TYPE = ["Adam", "AdamW"]
OPTIMIZER_PARAMS = {
    "Adam": {
        "weight_decay":  (0.00, "%0.2f"), # (default_value, accuracy)
    },
    "AdamW": {
        "weight_decay": (0.01, "%0.2f"),
    },
}
LOSS_TYPE = ["bce", "focal"]
LOSS_PARAMS = {
    "bce": {
        "pos_weight": (None, "%0.2f"),
    },
    "focal": {
        "alpha": (0.25, "%0.2f"),
        "gamma": (2.0, "%0.2f"),
    },
}
LR_SCHEDULER_TYPE = ["disable", "plateau"]
LR_SCHEDULER_PARAMS = {
    "disable": {

    },
    "plateau": {
        "factor": (0.1, "%0.1f"),
        "patience": (5, "%d"),
        "threshold": (1e-4, "%0.4f"),
        "min_lr": (1e-6, "%0.6f"),
        "eps": (1e-6, "%0.6f"),
    },
}
