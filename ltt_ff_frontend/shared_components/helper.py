from typing import Any

from loguru import logger

from ltt_ff_frontend.constant import BLANK_MODEL, INFERENCE_DEFAULT_RESULT_DIR, RESTRICT_OUTPUT_DIR


# put only codes like: format strings, aggregate data
# for generating figure, extract to one seperate component file


def format_model_name(name: str | None) -> str:
    if name is None:
        return "SCRATCH"
    if name == BLANK_MODEL:
        return BLANK_MODEL
    # For base model name (e.g. base/LTT_SW#x9u#N3#M0-M2#20250124T000000Z#55032dae#55032dae.encrypted.pth)
    if "/" in name:
        model_paths = name.split("/")
        model_type = model_paths[-2]
        model_name = model_paths[-1]
        return f"[{model_type}] {model_name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"
    # For output model name (e.g. 13feb_minye#x9u#tl#lg20250213T151435Z#55032dae#5fb1017f)
    else:
        return f"{name.replace('.encrypted', '').replace('.pth', '').replace('#', ' ')}"


def get_model_hash(name: str) -> str:
    if name.startswith("base"):
        return name.split("#", maxsplit=-1)[-1].split(".", maxsplit=1)[0]
    else:
        return name


def filter_recipe_columns(recipe: dict[str, Any]) -> dict[str, Any]:
    column_white_list = ["model_name", "threshold", "threshold_c", "threshold_d"]
    filtered_recipe = {"recipes": [{k: v for k, v in r.items() if k in column_white_list} for r in recipe["recipes"]]}

    return filtered_recipe


def is_valid_output_dir(output_dir: str) -> bool:
    """
    Disallow:
    - default output dir ("/mnt/dbpc/xxx")
    - directories not in /mnt/dbpc or /mnt/output
    - empty string ("")
    """
    # Block default output directory
    if output_dir == INFERENCE_DEFAULT_RESULT_DIR:
        logger.error(
            f"Default Result Directory detected ({INFERENCE_DEFAULT_RESULT_DIR}). "
            "Please enter an appropriate Result Directory."
        )
        return False

    # Block empty string
    if output_dir == "":
        logger.error("empty string detected.")
        return False

    # Block directories not in /mnt/dbpc or /mnt/output (for PROD)
    if RESTRICT_OUTPUT_DIR:
        allowed_directories = ("/mnt/dbpc", "/mnt/output")
        if not output_dir.startswith(allowed_directories):
            logger.error(f"Result Directory does not belong to one of the allowed directories: {allowed_directories}")
            return False
    return True
