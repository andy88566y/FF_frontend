def get_output_model_name(site: str, tool: str, tech_layer: str, layer_group: str) -> str:
    return "#".join([site, tool, tech_layer, layer_group])
