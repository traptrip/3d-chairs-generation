import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, no_update
from PIL import Image


def pil_image(image_filename):
    return Image.open(image_filename).resize((200, 200))


df = pd.read_csv("dataset_filtered.csv")  # .iloc[:100]
# columns: x,y,label,path

fig = go.Figure(
    data=[
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                color=df["label"],
                sizeref=15,
                sizemode="diameter",
                opacity=0.6,
            ),
        )
    ]
)

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig.update_traces(hoverinfo="none", hovertemplate=None)

fig.update_layout(
    width=1400,
    height=1000,
    plot_bgcolor="rgba(255,255,255,0.1)",
)

app = Dash("umap_data")

app.layout = html.Div(
    [
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ]
)


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = pil_image(df_row["image_path"])
    name = os.path.basename(df_row["image_path"])
    url = df_row["url"]

    children = [
        html.Div(
            [
                html.Img(src=img_src, style={"width": "100%"}),
                html.H3(f"{name}", style={"color": "black"}),
                html.P(url, style={"color": "darkblue"}),
            ],
            style={"width": "400px", "white-space": "normal"},
        )
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True, port=8083)
