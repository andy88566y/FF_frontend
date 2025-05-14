from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from ltt_ff_core.datamodel.training import DEFAULT_HYPER_PARAMS, HyperParams, ModelParams


class JobMetadata(BaseModel):
    """Fundamental job metadata data model.

    These fields are NOT required when the target is to update job metadata, so
    defaults are provided to prevent pydantic `ValidationError`.

    """

    status: Annotated[str, Field(description="Job status.", default="unknown")]
    progress: Annotated[int, Field(description="Job progress (%).", default=0)]
    start_time: Annotated[float, Field(description="Job start time.", default=-1)]
    end_time: Annotated[float, Field(description="Job start time.", default=-1)]
    message: Annotated[str, Field(description="Last updated/created message.", default="")]
    error_message: Annotated[str, Field(description="Error message.", default="")]

    model_config = ConfigDict(extra="forbid")


class TrainingJobMetadata(JobMetadata):
    """Training job metadata data model.

    The fields are highly overlapped with the `HyperParameters` data model as
    they are the same in esssence. However, these fields are NOT required when
    the target is to update job metadata. Under this context, defaults are provided
    to prevent pydantic `ValidationError`.

    """

    current_epoch: Annotated[int, Field(description="The current retraining epoch.", default=0)]
    site: Annotated[str, Field(description="The site.", default="")]
    tech_layer: Annotated[str, Field(description="The tech layer.", default="")]
    layer_group: Annotated[str, Field(description="The layer group.", default="")]
    training_history: Annotated[dict[str, list[Any]], Field(description="The training history.", default={})]
    # Essentially the same as TrainingRequestData
    base_model_name: Annotated[
        str | None, Field(description="The base model name to start finetune from.", default=None)
    ]
    model_params: Annotated[ModelParams | None, Field(description="Check out `ModelParams`", default=None)]
    output_model_name: Annotated[
        str,
        Field(description="The output model name formated as {site}#{tool}#{tech_layer}#{layer_group}.", default=""),
    ]
    multilot_config: Annotated[
        dict[str, Any],
        Field(
            description="Dict containing lot id, lrf path, and image dir for each group of training data.", default={}
        ),
    ]
    hyper_params: Annotated[
        HyperParams, Field(default=DEFAULT_HYPER_PARAMS, description="Model training hyper parameters.")
    ]

    model_config = ConfigDict(extra="forbid")


JobMetadataT = JobMetadata | TrainingJobMetadata
