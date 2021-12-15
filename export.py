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
    table_bits_per_key='total bits per key',
    point_lookup_percent='percentage of point queries')

file = "results.json" if len(sys.argv) < 2 else sys.argv[1]
with open(file) as data_file:
    data = json.load(data_file)

    # convert json results to dataframe
    df = pd.json_normalize(data, 'benchmarks')

    # augment additional computed columns
    # augment plotting datasets
    def magnitude(x):
        l = math.log(x, 10)
        rem = round(x/pow(10, l), 2)
        exp = int(round(l, 0))
        #return f'${rem} \cdot 10^{{{exp}}}$'
        return f'{rem}e-{exp}'
    df["method"] = df["label"].apply(lambda x : x.split(":")[0])
    df["dataset"] = df["label"].apply(lambda x : x.split(":")[1])
    df["elem_magnitude"] = df.apply(lambda x : magnitude(x["data_elem_count"]), axis=1)

    # prepare datasets for plotting & augment dataset specific columns
    lt_df = df[df["name"].str.lower().str.contains("probe")].copy(deep=True)
    ct_df = df[df["name"].str.lower().str.contains("construction")].copy(deep=True)
    mw_df = df[df["name"].str.lower().str.contains("mixed")].copy(deep=True)
    su_df = ct_df.copy(deep=True)

    # subset specific filtering & augmentation
    lt_df["probe_distribution"] = lt_df["label"].apply(lambda x : x.split(":")[2] if len(x.split(":")) > 2 else "-")
    lt_df["probe_size"] = lt_df["name"].apply(lambda x : int(x.split(",")[1].split(">")[0]))

    ct_df["cpu_time_per_key"] = ct_df.apply(lambda x : x["cpu_time"] / x["data_elem_count"], axis=1)
    ct_df["throughput"] = ct_df.apply(lambda x : 10**9/x["cpu_time_per_key"], axis=1)
    ct_df = ct_df[ct_df["data_elem_count"] > 9 * 10**7]

    mw_df["_sort_name"] = mw_df["label"].apply(lambda x : x.split(":")[0] if len(x.split(":")) > 0 else "-")
    mw_df["probe_distribution"] = mw_df["label"].apply(lambda x : x.split(":")[2] if len(x.split(":")) > 2 else "-")
    mw_df = mw_df.sort_values(["_sort_name", "point_lookup_percent"], ascending=True)

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

        # hide prefetched results by default
        fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
                   if trace.name.startswith("Prefetched") else ())

        return fig

    def plot_mixed():
        data = mw_df[mw_df["data_elem_count"] == 10**8]
        fig = px.line(
             data,
             x="point_lookup_percent",
             y="cpu_time",
             color="method",
             facet_row="probe_distribution",
             facet_col="dataset",
             category_orders={"dataset": ["seq", "gap_10", "uniform", "normal", "wiki", "osm", "fb"]},
             markers=True,
             log_x=False,
             labels=plot_labels,
             color_discrete_sequence=color_sequence,
             height=1000,
             title=f"Mixed workload - ns per key"
        )

        # hide prefetched results by default
        fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
                   if trace.name.startswith("Prefetched") else ())

        return fig


    def plot_construction_times():
        fig = px.bar(
            ct_df,
            x="elem_magnitude",
            y="throughput",
            color="method",
            barmode="group",
            facet_col="dataset",
            category_orders={"dataset": ["seq", "gap_10", "uniform", "normal", "wiki", "osm", "fb"]},
            labels=plot_labels,
            color_discrete_sequence=color_sequence,
            height=500,
            title=f"Construction time - keys per second"
            )

        # hide prefetched results by default
        fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
                   if trace.name.startswith("Prefetched") else ())

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

        # hide prefetched results by default
        fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
                   if trace.name.startswith("Prefetched") else ())

        return fig

    def plot_pareto_lookup_vs_space(probe_size):
        filtered = lt_df[(lt_df["probe_size"] == probe_size) & (lt_df["data_elem_count"] > 9 * 10**7)] 
        fig = px.scatter(
            filtered,
            x="cpu_time",
            y="table_bits_per_key",
            color="method",
            facet_row="probe_distribution",
            facet_col="dataset",
            category_orders={"dataset": ["seq", "gap_10", "uniform", "normal", "wiki", "osm", "fb"]},
            labels=plot_labels,
            color_discrete_sequence=color_sequence,
            height=1000,
            title=f"Pareto - lookup ({probe_size} elems in ns) vs space (total in bits/key)"
            )

        # hide prefetched results by default
        fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
                   if trace.name.startswith("Prefetched") else ())

        return fig

    outfile_name = "index.html" if len(sys.argv) < 4 else sys.argv[3]
    with open(f'{results_path}/{outfile_name}', 'w') as readme:
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

            {convert_to_html(plot_pareto_lookup_vs_space(0))}
            {convert_to_html(plot_pareto_lookup_vs_space(1))}
            {convert_to_html(plot_pareto_lookup_vs_space(10))}
            {convert_to_html(plot_pareto_lookup_vs_space(20))}

            {convert_to_html(plot_construction_times())}

            {convert_to_html(plot_mixed())}
          </body>
        </html>
        """))
