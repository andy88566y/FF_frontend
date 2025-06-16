import re


def get_output_model_name(site: str, tool: str, tech_layer: str, layer_group: str) -> str:
    # Assert none of the model name components contain symbols
    assert re.fullmatch(r"[A-Za-z0-9 ]*", f"{site}{tool}{tech_layer}{layer_group}") is not None, (
        "Site/Tool/TechLayer/LayerGroup cannot contain symbols!"
    )
    return "#".join([site, tool, tech_layer, layer_group])
