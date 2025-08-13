import matplotlib.pyplot as plt
import plotly.graph_objs as go
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm


OFFSETS = [
    {
        "x": -1,
        "y": 1,
    },
    {"x": 1, "y": 1},
    {"x": 0, "y": -1},
]


def plot_venn(
    sets: list[set],
    labels: list[str],
    width: int = 600,
    height: int = 600,
    colors: tuple[str] = ("#FF6F61", "#6B5B95", "#88B04B"),
    title: str = "Venn Diagram",
):
    v = venn3(
        sets,
        set_labels=labels,
        layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,) * 7),
    )
    plt.close()

    shapes = []
    annotations = []
    x_bounds, y_bounds = [], []

    for center, radius, set_label, color, offset in zip(v.centers, v.radii, v.set_labels, colors, OFFSETS):
        shapes.append(
            go.layout.Shape(
                type="circle",
                xref="x",
                yref="y",
                x0=center.x - radius,
                y0=center.y - radius,
                x1=center.x + radius,
                y1=center.y + radius,
                fillcolor=color,
                line_color=color,
                opacity=0.75,
            )
        )

        label_pos = set_label.get_position()
        annotations.append(
            go.layout.Annotation(
                xref="x",
                yref="y",
                x=label_pos[0] + offset["x"] * 0.05,
                y=label_pos[1] + offset["y"] * 0.05,
                text=set_label.get_text(),
                showarrow=False,
                font=dict(size=20, color="BLACK"),
            )
        )

        x_bounds.extend([center.x - radius, center.x + radius])
        y_bounds.extend([center.y - radius, center.y + radius])

    for i in range(7):
        if v.subset_labels[i] is not None:
            subset_pos = v.subset_labels[i].get_position()
            annotations.append(
                go.layout.Annotation(
                    xref="x",
                    yref="y",
                    x=subset_pos[0],
                    y=subset_pos[1],
                    text=v.subset_labels[i].get_text(),
                    showarrow=False,
                    font=dict(size=20, color="BLACK"),
                )
            )

    offset = 0.2
    x_range = [min(x_bounds) - offset, max(x_bounds) + offset]
    y_range = [min(y_bounds) - offset, max(y_bounds) + offset]

    fig = go.Figure()
    fig.update_xaxes(range=x_range, visible=False)
    fig.update_yaxes(range=y_range, scaleanchor="x", scaleratio=1, visible=False)

    fig.update_layout(
        margin=dict(b=0, l=10, pad=0, r=10, t=40),
        width=width,
        height=height,
        shapes=shapes,
        annotations=annotations,
        hovermode="closest",
        title=dict(text=title, x=0.5, xanchor="center"),
    )

    return fig
