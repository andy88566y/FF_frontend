import pandas as pd

def generate_lttswadc_html(lttswadc_map):
    indices = []
    for i in range(0, len(lttswadc_map), 8):
        index_row = []
        for col_offset, j in enumerate(range(i, min(i + 8, len(lttswadc_map)))):
            edge_class = "left-edge" if col_offset == 0 else ""
            if lttswadc_map[j][1] == 1:
                cell_html = f"""<div class='tooltip {edge_class}'><span style='color:red'>{j}</span><span class='tooltiptext'>{lttswadc_map[j][0]}</span></div>"""
            else:
                cell_html = f"""<div class='tooltip {edge_class}'>{j}<span class='tooltiptext'>{lttswadc_map[j][0]}</span></div>"""
            index_row.append(cell_html)
        indices.append(index_row)

    lrf_constant_df = pd.DataFrame(indices)
    return render_lttswadc_table(lrf_constant_df.to_html(escape=False, index=False, header=False))


def render_lttswadc_table(html_table: str) -> str:
    return f"""
    <div style="max-width: 100%; overflow-x: auto;">
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
            }}
            th, td {{
                text-align: center;
                padding: 6px;
                border: 1px solid #ccc;
                word-wrap: break-word;
                font-size: 14px;
                overflow: visible;
            }}
            .tooltip {{
                position: relative;
                display: inline-block;
                cursor: pointer;
                overflow: visible;
            }}
            .tooltip .tooltiptext {{
                visibility: hidden;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
                white-space: nowrap;
                margin-top: 6px;
            }}
            .tooltip.left-edge .tooltiptext {{
                left: 0;
                transform: none;
            }}
            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
        </style>
        {html_table}
    </div>
    """
