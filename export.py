import json
import math
import pandas as pd
import sys
import git
from pathlib import Path
from inspect import cleandoc

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly
from pretty_html_table import build_table

# plot colors
pal = px.colors.qualitative.Plotly
color_sequence = ["#BBB", "#777", "#111", pal[9], pal[4], pal[6], pal[1], pal[0], "#58a2c4", pal[5], pal[2], pal[7], pal[8], pal[3]]

# plot labels
plot_labels = dict(
    cpu_time='ns per key',
    data_elem_count='dataset size',
    table_bits_per_key='total bits per key')


file = "benchmark_results.json" if len(sys.argv) < 2 else sys.argv[1]
with open(file) as data_file:
    data = json.load(data_file)

    # convert json results to dataframe
    df = pd.json_normalize(data, 'benchmarks')

    # augment additional computed columns
    df["method"] = df["label"].apply(lambda x : x.split(":")[0])
    df["dataset"] = df["label"].apply(lambda x : x.split(":")[1])
    df["probe_distribution"] = df["label"].apply(lambda x : x.split(":")[2] if len(x.split(":")) > 2 else "-")
    df["probe_size"] = df["name"].apply(lambda x : int(x.split(",")[1].split(">")[0]))

    # prepare datasets for plotting & augment dataset specific columns
    lt_df = df.copy(deep=True)
    su_df = df.copy(deep=True)

    su_df = su_df[(su_df["probe_size"] == 0) & (su_df["probe_distribution"].str.match("uniform"))]

    # ensure export output folder exists
    results_path = "docs" if len(sys.argv) < 3 else sys.argv[2]
    Path(results_path).mkdir(parents=True, exist_ok=True)

    def convert_to_html(fig):
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def plot_lookup_times(probe_size):
        data = lt_df[lt_df["probe_size"] == probe_size]
        fig = px.line(
            data,
            x="data_elem_count",
            y="cpu_time",
            color="method",
            facet_row="probe_distribution",
            facet_col="dataset",
            category_orders={"dataset": ["seq", "gap_10", "uniform", "normal", "wiki", "osm", "fb"]},
            markers=True,
            log_x=True,
            labels=plot_labels,
            color_discrete_sequence=color_sequence,
            height=1000,
            title=f"Probe (size: {probe_size}) - ns per key"
            )
        return fig

    def plot_space_usage():
        fig = px.line(
            su_df,
            x="data_elem_count",
            y="table_bits_per_key",
            color="method",
            facet_col="dataset",
            category_orders={"dataset": ["seq", "gap_10", "uniform", "normal", "wiki", "osm", "fb"]},
            markers=True,
            log_x=True,
            labels=plot_labels,
            color_discrete_sequence=color_sequence,
            height=500,
            title=f"Total space usage - bits per key"
            )
        return fig


    with open(f'{results_path}/index.html', 'w') as readme:
        readme.write(cleandoc(f"""
        <!doctype html>
        <html>
          <head>
              <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
          </head>

          <body>
            {convert_to_html(plot_lookup_times(0))}
            {convert_to_html(plot_lookup_times(1))}
            {convert_to_html(plot_lookup_times(10))}
            {convert_to_html(plot_lookup_times(20))}

            {convert_to_html(plot_space_usage())}
          </body>
        </html>
        """))
