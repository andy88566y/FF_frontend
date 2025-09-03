import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from loguru import logger
from matplotlib.axes import Axes

from ltt_ff_frontend.helpers import api_helper
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# TODO: Change to Plotly
def draw_diff_img(data_yaml_path: str, lot_id: str, defect_id: str, norm: bool, diff_clip: int = 0.3):
    data_lots = api_helper.list_yaml_lots(data_yaml_path)
    diff_img_data = api_helper.generate_diff_images(data_yaml_path, lot_id, defect_id, norm)

    if diff_img_data["status"] == "error":
        st.error(f"Error: {diff_img_data['message']}")
        return

    defect_info = diff_img_data["defect_meta"]
    image_data = diff_img_data["image_data"]
    layer_name = data_lots[lot_id].get("layer_group", "")

    tensor_keys = ["aligned_ref", "aligned_test", "aligned_diff", "feature_map"]
    for p in ["Rt", "T"]:
        for k in image_data[p]:
            if k in tensor_keys:
                image_data[p][k] = np.array(image_data[p][k])

    def set_ticks(ax: Axes) -> None:
        # ax.set_axis_off()
        ax.set_xticks(np.linspace(0, 256, num=8 + 1))
        ax.set_yticks(np.linspace(0, 256, num=8 + 1))

    logger.debug(f"[{lot_id}:{defect_id}] Generating diff image...")

    fig, axes = plt.subplots(2, 4, width_ratios=(0.8, 0.8, 1.0, 0.8), figsize=(18, 10))
    fig.suptitle(
        f"[Defect View] {layer_name} Lot: {lot_id} | Defect ID: {defect_id}"
        f" | X: {defect_info['X']} | Y: {defect_info['Y']}\n"
        f"ClassType: {defect_info['ClassType']}, isDefect: {defect_info['isDefect']},"
        f" ParticleModeOnly: {defect_info['particleModeOnly']} {defect_info.get('relaxedParticleMode', False)} "
        f"{defect_info.get('ulParticleMode', False)} {image_data['avail_refs'] == 0}"
        f" (lrf-ORIG | pc: {defect_info['PixelCount']}, h: {defect_info['H']}, w: {defect_info['W']})",
        fontsize=14,
    )

    for rid, ptype in enumerate(["Rt", "T"]):
        axes[rid, 0].imshow(image_data[ptype]["aligned_ref"], cmap="gray")
        if len(image_data["avail_refs"]) == 1:
            axes[rid, 0].set_title(f"Reference [{ptype}]\n(Available Refs: {image_data['avail_refs'][0]})")
        else:
            axes[rid, 0].set_title(
                f"Reference [{ptype}]\n(Available Refs: Median of {['C'] + image_data['avail_refs']})"
            )
        set_ticks(axes[rid, 0])
        axes[rid, 1].imshow(image_data[ptype]["aligned_test"], cmap="gray")
        axes[rid, 1].set_title(
            f"Test ({image_data['worst_test']}) [{ptype}]\n"
            f"(best_pos: {[round(float(x), 2) for x in image_data[ptype]['best_pos']]})"
        )
        set_ticks(axes[rid, 1])
        # Colormaps: seismic, coolwarm, viridis
        rt_diff_img = axes[rid, 2].imshow(
            image_data[ptype]["aligned_diff"], cmap="seismic", vmin=-diff_clip, vmax=diff_clip
        )
        axes[rid, 2].set_title(f"Difference [{ptype}]\n(max_diff: {round(image_data[ptype]['max_diff'], 3)})")
        set_ticks(axes[rid, 2])
        fig.colorbar(rt_diff_img, ax=axes[rid, 2], location="right", shrink=0.7, fraction=0.15, pad=0.05)

        pt_mapping = {
            -1: "UNK",
            -2: "H1D",
            -3: "V1D",
            -4: "2D",
        }

        ft_color_mapping = {
            0: ("UNK", "white"),
            1: ("clear", "purple"),
            2: ("opaque", "blue"),
            3: ("edge", "yellow"),
            4: ("non_edge", "lightsteelblue"),
            5: ("defect_edge", "orange"),
            6: ("pin_defect", "red"),
        }

        cmap = mcolors.ListedColormap([x[1] for x in ft_color_mapping.values()])
        norm = mcolors.BoundaryNorm(list(range(6 + 1 + 1)), cmap.N)

        axes[rid, 3].imshow(image_data[ptype]["feature_map"], cmap=cmap, norm=norm)
        axes[rid, 3].set_title(f"Features [{ptype}]\nPattern [{pt_mapping[image_data['pattern_type']]}]")
        set_ticks(axes[rid, 3])

        max_pos = np.unravel_index(image_data[ptype]["aligned_diff"].argmax(), image_data[ptype]["aligned_diff"].shape)
        min_pos = np.unravel_index(image_data[ptype]["aligned_diff"].argmin(), image_data[ptype]["aligned_diff"].shape)

        crop_size = 16
        for ax_idx in range(4):
            # It is known that due to edges, MATRICS_DefectPoint is not always on 128, 128.
            # TODO: Dynamic crop size according to lrf info (H, W, PixelSize)
            # Tool Defect position (green box)
            rect_tool = plt.Rectangle(
                (
                    image_data[ptype]["tool_defect_loc"][0] + round(image_data[ptype]["best_pos"][1]) - 16 - crop_size,
                    image_data[ptype]["tool_defect_loc"][1] + round(image_data[ptype]["best_pos"][0]) - 16 - crop_size,
                ),
                crop_size * 2,
                crop_size * 2,
                fill=False,
                color="green",
                linewidth=0.5,
            )
            axes[rid, ax_idx].add_patch(rect_tool)

            # Max position (red box)
            rect_max = plt.Rectangle(
                (max_pos[1] - crop_size, max_pos[0] - crop_size),
                crop_size * 2,
                crop_size * 2,
                fill=False,
                color="red",
                linewidth=1,
            )
            axes[rid, ax_idx].add_patch(rect_max)

            # Min position (blue box)
            rect_min = plt.Rectangle(
                (min_pos[1] - crop_size, min_pos[0] - crop_size),
                crop_size * 2,
                crop_size * 2,
                fill=False,
                color="blue",
                linewidth=1,
            )
            axes[rid, ax_idx].add_patch(rect_min)

    fig.tight_layout(pad=1.5, h_pad=2.2, w_pad=2.2)

    st.pyplot(fig)


def draw_diff_img_plotly(data_yaml_path: str, lot_id: str, defect_id: str, norm: bool, diff_clip: float = 0.3):
    # Load data
    diff_img_data = api_helper.generate_diff_images(data_yaml_path, lot_id, defect_id, norm)

    if diff_img_data["status"] == "error":
        st.error(f"Error: {diff_img_data['message']}")
        return

    # Extract metadata and image tensors
    defect_info = diff_img_data["defect_meta"]
    image_data = diff_img_data["image_data"]

    title1 = None
    if len(image_data["avail_refs"]) == 1:
        title1 = (f"(Available Refs: {image_data['avail_refs'][0]})")
    else:
        title1 = f"(Available Refs: Median of {['C'] + image_data['avail_refs']})"
    title2_Rt = f"Test ({image_data['worst_test']}) [Rt]<br>" + f"(best_pos: {[round(float(x), 2) for x in image_data['Rt']['best_pos']]})"
    title2_T = f"Test ({image_data['worst_test']}) [T]<br>" + f"(best_pos: {[round(float(x), 2) for x in image_data['T']['best_pos']]})"
    title3_Rt = f"Difference [Rt]<br>(max_diff: {round(image_data['Rt']['max_diff'], 3)})"
    title3_T = f"Difference [T]<br>(max_diff: {round(image_data['T']['max_diff'], 3)})"
    pt_mapping = {
            -1: "UNK",
            -2: "H1D",
            -3: "V1D",
            -4: "2D",
        }
    title4_Rt = f"Features [Rt]\nPattern [{pt_mapping[image_data['pattern_type']]}]"
    title4_T = f"Features [T]\nPattern [{pt_mapping[image_data['pattern_type']]}]"

    # Create Plotly subplots
    fig = make_subplots(rows=2, cols=4, subplot_titles=[
        "Reference [Rt]<br>"+title1, title2_Rt, title3_Rt, title4_Rt,
        "Reference [T]<br>"+title1, title2_T, title3_T, title4_T
    ], horizontal_spacing=0.1, vertical_spacing=0.05)
    
    fig.update_layout(
        font=dict(size=10),  # Smaller font for all text including subplot titles
    )


    # Convert image tensors to numpy arrays
    for p in ["Rt", "T"]:
        for k in ["aligned_ref", "aligned_test", "aligned_diff", "feature_map"]:
            image_data[p][k] = np.array(image_data[p][k])
    
    ft_color_mapping = {
        0: ("UNK", "white"),
        1: ("clear", "purple"),
        2: ("opaque", "blue"),
        3: ("edge", "yellow"),
        4: ("non_edge", "lightsteelblue"),
        5: ("defect_edge", "orange"),
        6: ("pin_defect", "red"),
    }
    
    feature_colorscale = [[i / 6, color] for i, (_, color) in enumerate(ft_color_mapping.values())]

    seismic_colorscale = [
        [0.0, "blue"],
        [0.5, "white"],
        [1.0, "red"]
    ]

    # Loop through Rt and T images
    for rid, ptype in enumerate(["Rt", "T"]):
        ref_img = image_data[ptype]["aligned_ref"]
        test_img = image_data[ptype]["aligned_test"]
        diff_img = image_data[ptype]["aligned_diff"]     
        feature_map = image_data[ptype]["feature_map"]

        max_pos = np.unravel_index(diff_img.argmax(), diff_img.shape)
        min_pos = np.unravel_index(diff_img.argmin(), diff_img.shape)
        crop_size = 16

        
        for cid, img, cmap, vmin, vmax, show_scale in zip(
            [1, 2, 3, 4],
            [ref_img, test_img, diff_img, feature_map],
            ["gray", "gray", seismic_colorscale, feature_colorscale],
            [None, None, -diff_clip, 0],
            [None, None, diff_clip, 6],
            [False, False, True, False]
        ):
            fig.add_trace(go.Heatmap(
                z=img,
                colorscale=cmap,
                zmin=vmin,
                zmax=vmax,
                showscale=show_scale,
                hoverinfo="skip"
            ), row=rid + 1, col=cid)

            # Add rectangles for defect positions
            tool_x = image_data[ptype]["tool_defect_loc"][0] + round(image_data[ptype]["best_pos"][1]) - 16
            tool_y = image_data[ptype]["tool_defect_loc"][1] + round(image_data[ptype]["best_pos"][0]) - 16

            for pos, color in [(max_pos, "red"), (min_pos, "blue"), ((tool_y, tool_x), "green")]:
                fig.add_shape(type="rect",
                    x0=pos[1] - crop_size, y0=pos[0] - crop_size,
                    x1=pos[1] + crop_size, y1=pos[0] + crop_size,
                    line=dict(color=color, width=1),
                    row=rid + 1, col=cid
                )


    # Update layout
    fig.update_layout(
        title_text=(
            f"[Defect Aligned Comparison] Lot: {lot_id} | Defect ID: {defect_id} | "
            f"X: {defect_info['X']} | Y: {defect_info['Y']}<br>"
            f"ClassType: {defect_info['ClassType']}, isDefect: {defect_info['isDefect']}, "
            f" ParticleModeOnly: {defect_info['particleModeOnly']} {defect_info.get('relaxedParticleMode', False)} "
            f"{defect_info.get('ulParticleMode', False)} {image_data['avail_refs'] == 0}"
            f" (lrf-ORIG | pc: {defect_info['PixelCount']}, h: {defect_info['H']}, w: {defect_info['W']})"
        ),
        title_y=0.98,  # Move title closer to top
        height=800,
        width=1100,
        autosize=False,
    )
    fig.update_yaxes(autorange='reversed')

    for i in range(1, 9):  # 8 subplots
        fig.update_yaxes(scaleanchor=f"x{i}", row=(i - 1) // 4 + 1, col=(i - 1) % 4 + 1)
    fig.update_xaxes(ticks="outside", ticklen=3)
    fig.update_yaxes(ticks="outside", ticklen=3)

    fig.update_yaxes(constrain='domain')
    fig.update_xaxes(constrain='domain')

    st.plotly_chart(fig)


def app() -> None:
    st.title("Defect Viewer")

    default_data_yaml_path = "/mnt/dbpc/FalseFilterDataSet/WeeklyYaml/Wxxx_data_2025xxxx.yaml"
    data_yaml_path = st.text_input("Data Yaml Path", default_data_yaml_path)
    if data_yaml_path == default_data_yaml_path:
        return

    data_lots = api_helper.list_yaml_lots(data_yaml_path)

    r1_col1, r1_col2, r1_col3 = st.columns([3, 3, 1])

    with r1_col1:
        lot_id = st.selectbox("Lot ID", [""] + list(data_lots.keys()))
    with r1_col2:
        defect_id = st.text_input("Defect ID")
    with r1_col3:
        norm = st.toggle("Normalize", value=True)

    if lot_id != "" and defect_id != "":
        draw_diff_img(data_yaml_path, lot_id, defect_id, norm, diff_clip=0.3)
