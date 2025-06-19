from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class ModelParams(BaseModel):
    model_type: Annotated[
        str,
        Field(default="DualStreamCNN", description="The type of the model to use. Valid options: [DualStreamCNN]"),
    ]
    model_init_params: Annotated[
        dict[str, Any],
        Field(default={}, description="Initialization parameters for the specific type of model."),
    ]
    model_threshold: Annotated[float, Field(default=0.1, description="The pipeline1 threshold.")]
    model_threshold_c: Annotated[float, Field(default=0.1, description="The pipeline2 threshold.")]

    model_config = ConfigDict(extra="forbid")


class HyperParams(BaseModel):
    optimizer_type: Annotated[str, Field(description="Name of optimizer (e.g. Adam, AdamW)")]
    optimizer_params: Annotated[
        dict[str, Any] | None,
        Field(description="Parameters required for the selected optimizer type."),
    ]
    loss_type: Annotated[str, Field(description="Name of loss type (e.g. bce, focal)")]
    loss_params: Annotated[dict[str, Any] | None, Field(description="Parameters required for the selected loss type.")]
    lr_scheduler_type: Annotated[str, Field(description="Name of lr scheduler (e.g. disable, plateau)")]
    lr_scheduler_params: Annotated[
        dict[str, Any] | None, Field(description="Parameters required for the selected lr scheduler type.")
    ]
    sampler_params: Annotated[
        dict[str, Any] | None,
        Field(description="Parameters required for the selected sampler type."),
    ]
    batch_size: Annotated[int, Field(default=32, description="Training batch size.")]
    epochs: Annotated[int, Field(default=30, description="Number of training epochs.")]
    learning_rate: Annotated[
        float,
        Field(
            default=0.001,
            description="Learning rate.",
        ),
    ]
    tool: Annotated[str, Field(default="x9u", description="Training tool.")]

    model_config = ConfigDict(extra="forbid")


class FFCoreTrainingRequest(BaseModel):
    base_model_name: Annotated[str | None, Field(description="The base model name to start finetune from.")]
    model_params: Annotated[ModelParams | None, Field(description="Model-related parameters.")]
    output_model_name: Annotated[
        str,
        Field(description="The output model name formated as {site}#{tool}#{tech_layer}#{layer_group}."),
    ]
    multilot_config: Annotated[
        dict[str, Any],
        Field(description="Dict containing lot id, lrf path, and image dir for each group of training data."),
    ]
    hyper_params: Annotated[HyperParams, Field(description="Model training hyper parameters.")]
    skip_particle_mode_defects: Annotated[
        bool, Field(default=True, description="Flag to enable/disable training using particle mode defects.")
    ]

    model_config = ConfigDict(extra="forbid")
