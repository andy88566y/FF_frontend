API_ROOT = "http://localhost:6500/api/v1/"
TIMEOUT = 10
OPTIMIZER_TYPE = ["Adam", "AdamW"]
OPTIMIZER_PARAMS = {
    "Adam": {

    },
    "AdamW": {
        "weight_decay": 0.01,
    },
}
LOSS_TYPE = ["bce", "focal"]
LOSS_PARAMS = {
    "bce": {

    },
    "focal": {
        "alpha": 0.25,
        "gamma": 2.0,
    },
}
LR_SCHEDULER_TYPE = ["disable", "plateau"]
LR_SCHEDULER_PARAMS = {
    "disable": {

    },
    "plateau": {
        "factor": 0.1,
        "patience": 5,
        "threshold": 1e-4,
        "min_lr": 1e-6,
        "eps": 1e-6,
    },
}
