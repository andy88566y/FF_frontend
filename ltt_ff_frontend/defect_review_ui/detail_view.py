import os
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots


def load_image(path: str) -> Image:
    if path and os.path.exists(path):
        return Image.open(path).convert("RGBA")
    else:
        width, height = 200, 200
        image = Image.new('RGBA', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        draw.line((0, 0, width, height), fill='red', width=10)
        draw.line((0, height, width, 0), fill='red', width=10)
        return image

def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def add_shape_to_fig(fig: go.Figure, shape_type: str, x0: float, y0: float, x1: float, y1: float, title: str) -> None:
    if fig is not None:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="red", width=3))
        fig.update_layout(
            title={"text": title, "y": 1, "x": 0.5, "xanchor": "center", "yanchor": "top"}, margin=dict(t=30)
        )


def get_pixel_values(
    images: list[Image], selection: dict[str, Any], direction: str, fig2: go.Figure, fig3: go.Figure
) -> list[np.ndarray]:
    x0, x1 = selection["box"][-1]["x"][0], selection["box"][-1]["x"][1]
    y0, y1 = selection["box"][-1]["y"][0], selection["box"][-1]["y"][1]
    pixel_values = []

    if direction == "horizontal":
        y_midpoint = (y0 + y1) / 2
        for img in images:
            img_array = np.array(img)
            row_values = img_array[int(y_midpoint), int(x0) : int(x1)]
            pixel_values.append(row_values)
        add_shape_to_fig(fig2, "line", x0, y_midpoint, x1, y_midpoint, "Line Pos (RT)")
        add_shape_to_fig(fig3, "line", x0, y_midpoint, x1, y_midpoint, "Line Pos (T)")

    elif direction == "vertical":
        x_midpoint = (x0 + x1) / 2
        for img in images:
            img_array = np.array(img)
            col_values = img_array[int(y1) : int(y0), int(x_midpoint)]
            pixel_values.append(col_values[::-1])
        add_shape_to_fig(fig2, "line", x_midpoint, y0, x_midpoint, y1, "Line Pos (RT)")
        add_shape_to_fig(fig3, "line", x_midpoint, y0, x_midpoint, y1, "Line Pos (T)")

    elif direction == "right_diagonal":
        points = bresenham_line(int(x0), int(y1), int(x1), int(y0))
        for img in images:
            img_array = np.array(img)
            diag_values = [img_array[y][x] for x, y in points]
            pixel_values.append(np.vstack(diag_values))
        add_shape_to_fig(fig2, "line", x0, y1, x1, y0, "Line Pos (RT)")
        add_shape_to_fig(fig3, "line", x0, y1, x1, y0, "Line Pos (T)")

    elif direction == "left_diagonal":
        points = bresenham_line(int(x0), int(y0), int(x1), int(y1))
        for img in images:
            img_array = np.array(img)
            diag_values = [img_array[y][x] for x, y in points]
            pixel_values.append(np.vstack(diag_values))
        add_shape_to_fig(fig2, "line", x0, y0, x1, y1, "Line Pos (RT)")
        add_shape_to_fig(fig3, "line", x0, y0, x1, y1, "Line Pos (T)")

    return pixel_values


def create_figure(image: Image) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Image(z=np.array(image)))
    fig.update_layout(
        width=100,
        height=150,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        dragmode="zoom",
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, range=[0, 100], autorange=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, 100], autorange=True)
    return fig


def create_chart(df: pd.DataFrame, domain: list[str], range_colors: list[str]) -> alt.Chart:
    chart = (
        alt.Chart(df.reset_index().melt("index", var_name="Line", value_name="value"))
        .mark_line()
        .encode(
            x="index:Q",
            y="value:Q",
            color=alt.Color(
                "Line:N", scale=alt.Scale(domain=domain, range=range_colors), legend=alt.Legend(orient="bottom")
            ),
        )
        .properties(width=700, height=300)
        .configure_axisX(domain=False, title=None, labels=False)
    )
    return chart

# ext= "lrf" or "blrf"
def app(selected_row: pd.DataFrame, image_dir: str, ext: str = 'lrf') -> None:
    st.subheader(f"Defect No {selected_row['No'].values[0]}, UniqueID: {selected_row['UniqueID'].values[0]}")

    # Create a single row with three columns for X, Y, and ClassType
    cola, colb, colc = st.columns(3)
    with cola:
        st.write(f"**X:** {selected_row.X.values[0]}")
    with colb:
        st.write(f"**Y:** {selected_row.Y.values[0]}")
    with colc:
        class_type_int = int(selected_row.ClassType.values[0])
        st.write(f"**Class Type:** {class_type_int}")

    st.subheader("InstantReview", divider="gray")

    # Create three columns
    col1, col2, col3 = st.columns(3)

    type_options = ["L", "L_p", "U", "U_p", "_M", "_L", "_U"]
    ref_image_path, diff_image_path = None, None
    # Construct the file paths_Rt
    if ext == 'blrf':
        # .blrf
        base_path = f"{image_dir}/Images/InstantReviewTDI_2/"
        no = selected_row.UniqueID.values[0]
        test_image_path = f"{base_path}{no}_C.png"
        for type_option in ["_M_D", "_U_D", "_L_D"]:
            potential_path = f"{base_path}{no}{type_option}.png"
            if os.path.exists(potential_path):
                diff_image_path = potential_path
                break
    elif ext == 'lrf':
        # .lrf
        base_path = f"{image_dir}/Images/InstantReviewRt/"
        no = selected_row.No.values[0]
        test_image_path = f"{base_path}{no}.png"
        diff_image_path = f"{base_path}{no}D.png"
    else:
        logger.error('unkown format when infering image file name.')

    # Find the correct test image path
    for type_option in type_options:
        potential_path = f"{base_path}{no}{type_option}.png"
        if os.path.exists(potential_path):
            ref_image_path = potential_path
            break

    # Construct the file paths_T
    ref_image_path_T, diff_image_path_T = None, None
    if ext == 'blrf':
        # .blrf
        base_path_T = f"{image_dir}/Images/InstantReviewTDI_1/"
        test_image_path_T = f"{base_path_T}{no}_C.png"
        for type_option in ["_M_D", "_U_D", "_L_D"]:
            potential_path = f"{base_path}{no}{type_option}.png"
            if os.path.exists(potential_path):
                diff_image_path_T = potential_path
                break
    elif ext == 'lrf':
        # .lrf
        base_path_T = f"{image_dir}/Images/InstantReviewT/"
        test_image_path_T = f"{base_path_T}{no}.png"
        diff_image_path_T = f"{base_path_T}{no}D.png"
    else:
        logger.error('unkown format when infering image file name.')
    # Find the correct test image path_T
    for type_option in type_options:
        potential_path_T = f"{base_path_T}{no}{type_option}.png"
        if os.path.exists(potential_path_T):
            ref_image_path_T = potential_path_T
            break

    # Load images
    ref_image = load_image(ref_image_path)
    test_image = load_image(test_image_path)
    diff_image = load_image(diff_image_path)
    ref_image_T = load_image(ref_image_path_T)
    test_image_T = load_image(test_image_path_T)
    diff_image_T = load_image(diff_image_path_T)
    images = [ref_image, test_image, diff_image, ref_image_T, test_image_T, diff_image_T]

    # Create a subplot with shared axes
    # TODO: Movement along the Y-axis will update synchronously, but movement along the X-axis will not.
    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Reference Image Rt",
            "Test Image Rt",
            "Difference Rt",
            "Reference Image T",
            "Test Image T",
            "Difference T",
        ),
    )

    # Add images to the subplot
    fig.add_trace(go.Image(z=np.array(ref_image)), row=1, col=1)
    fig.add_trace(go.Image(z=np.array(test_image)), row=1, col=2)
    fig.add_trace(go.Image(z=np.array(diff_image)), row=1, col=3)
    fig.add_trace(go.Image(z=np.array(ref_image_T)), row=2, col=1)
    fig.add_trace(go.Image(z=np.array(test_image_T)), row=2, col=2)
    fig.add_trace(go.Image(z=np.array(diff_image_T)), row=2, col=3)

    # Add a scatter trace
    fig.add_trace(go.Scatter(x=[0, 200 * 1], y=[0, 200 * 1], mode="markers", marker_opacity=0), row=1, col=1)

    # Update layout
    fig.update_layout(
        width=900,
        height=800,
        margin={"t": 20, "b": 20, "l": 20, "r": 20},
        modebar={
            "orientation": "v",
            "bgcolor": "rgba(0,0,0,0)",
            "color": "gray",
            "activecolor": "#c37969",
        },
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        dragmode="zoom",
        modebar_add=["select"],
    )

    # Update axes to fix the image size
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        range=[0, 200],
        matches="y",
    )
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        range=[0, 200],
        matches="x",
    )
    # # Autoscale the images
    # fig.update_xaxes(autorange=True)
    # fig.update_yaxes(autorange=True)

    # Create figures for the test images
    fig2 = create_figure(test_image)
    fig3 = create_figure(test_image_T)

    # show image and enable selection
    event = st.plotly_chart(fig, use_container_width=True, key="images", on_select="rerun")

    # Ensure direction is set in session state
    if "direction" not in st.session_state:
        st.session_state.direction = "horizontal"

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Horizontal", icon=":material/east:", key="horizontal_button", use_container_width=True):
            st.session_state.direction = "horizontal"
    with col2:
        if st.button("Vertical", icon=":material/south:", key="vertical_button", use_container_width=True):
            st.session_state.direction = "vertical"
    with col3:
        if st.button(
            "Downward Diagonal", icon=":material/south_east:", key="left_diagonal_button", use_container_width=True
        ):
            st.session_state.direction = "left_diagonal"
    with col4:
        if st.button(
            "Upward Diagonal", icon=":material/north_east:", key="right_diagonal_button", use_container_width=True
        ):
            st.session_state.direction = "right_diagonal"

    # Get the selection result
    try:
        selection = event.selection

        if selection and "box" in selection and len(selection["box"]) > 0:
            pixel_values = get_pixel_values(images, selection, st.session_state.direction, fig2, fig3)
            extracted_data = {index: array[:, 0] for index, array in enumerate(pixel_values)}

            # data change to dataFrame
            df_pixel = pd.DataFrame(extracted_data)
            df_pixel.columns = [
                "Reference Image Rt",
                "Test Image Rt",
                "Difference Rt",
                "Reference Image T",
                "Test Image T",
                "Difference T",
            ]

            # split to 2 charts
            df_pixel_1 = df_pixel.iloc[:, :3]
            df_pixel_2 = df_pixel.iloc[:, 3:]

            # Configuration to remove the modebar
            config = {"displayModeBar": False}

            # Plot line charts
            col11, col12 = st.columns([1, 1])
            with col11:
                chart_1 = create_chart(df_pixel_1, ["Reference Image Rt", "Test Image Rt"], ["darkblue", "lightblue"])
                col111, col112 = st.columns([4, 1])
                with col111:
                    st.altair_chart(chart_1)
                with col112:
                    st.plotly_chart(fig2, use_container_width=True, key="images2", config=config)

            with col12:
                chart_2 = create_chart(df_pixel_2, ["Reference Image T", "Test Image T"], ["darkblue", "lightblue"])
                col121, col122 = st.columns([5, 1])
                with col121:
                    st.altair_chart(chart_2)
                with col122:
                    st.plotly_chart(fig3, use_container_width=True, key="images3", config=config)

    except KeyError as e:
        st.write(f"KeyError: {e}. Please make a valid selection.")


# def overlay_images(base_image, overlay_image, alpha=0.5):
#     return Image.blend(base_image, overlay_image, alpha)

#     st.header("Overlayed Image Display")
#     # choose
#     show_image1 = st.checkbox('Show Reference Image', value=True)
#     show_image2 = st.checkbox('Show Test Image', value=True)
#     show_image3 = st.checkbox('Show Difference Image', value=True)

#     # init
#     base_image = Image.new('RGBA', ref_image.size)

#     # overlay
#     if show_image1:
#         base_image = overlay_images(base_image, ref_image, alpha=0.7)
#     if show_image2:
#         base_image = overlay_images(base_image, test_image, alpha=0.7)
#     if show_image3:
#         base_image = overlay_images(base_image, diff_image, alpha=0.7)

#     # display overlay
#     fig_overlay = px.imshow(np.array(base_image))
#     fig_overlay.update_layout(title="Overlayed Image")
#     st.plotly_chart(fig_overlay, use_container_width=True)
