#!/bin/python

import argparse
import torchfile
import numpy as np
import pandas as pd
import os
from colour import Color
import logging
from plotly import tools
import plotly.graph_objs as go
from plotly import offline

logger = logging.getLogger(__name__)

layout = dict(
    paper_bgcolor = 'rgb(255,255,255)',
    plot_bgcolor = 'rgb(229,229,229)',
    font = dict(family='Courier New, monospace', size=18),
    xaxis=dict(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        title='Episodes',
        zeroline=False,
        anchor='y',
        autorange=True
    ),
    xaxis2=dict(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        title='Episodes',
        zeroline=False,
        anchor='free',
        autorange=True
    ),
    xaxis3=dict(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        title='Episodes',
        zeroline=False,
        anchor='free',
        autorange=True
    ),
    yaxis=dict(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        title='(100 episode) Rewards',
        zeroline=False,
        anchor='x',
        domain=[0, 1.0],
        autorange=True
    ),
    yaxis2=dict(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        title='Training Loss',
        zeroline=False,
        anchor='x2',
        domain=[0, 1.0],
        autorange=True
    ),
    yaxis3=dict(
        gridcolor='rgb(255,255,255)',
        showgrid=True,
        showline=False,
        showticklabels=True,
        tickcolor='rgb(127,127,127)',
        ticks='outside',
        title='V-function estimate',
        zeroline=False,
        anchor='x3',
        domain=[0, 1.0],
        autorange=True
    ),
    titlefont=dict(size=20),
    showlegend = True,
    legend=dict(
        x=0.86,
        y=0.98,
        traceorder='normal',
    ),
    hovermode='x',
    autosize = True,
)

def parse_args(optional_string=""):
    parser = argparse.ArgumentParser("Plot comparision graph")

    parser.add_argument("--experiments", type=str, default="/home/rchouras/workplace/rishav-internship/code/Atari-experiments/experiments", required=False, help="directory to store your experiments in")
    parser.add_argument("--environment", type=str, default="rlenvs.Catch", required=False, help="The environment to compare on")
    parser.add_argument("--smoothing", type=int, default=100, required=False, help="smoothing window for the rewards plots")
    parser.add_argument("--color", type=dict, default=('yellow', 'purple'), required=False, help="colors of the algorithms")

    if optional_string:
        return parser.parse_args(optional_string)

    return parser.parse_args()


def uniform_scores(filename, smoothing=1):
    scores = torchfile.load(filename=filename)
    regularized_scores = pd.Series(np.array(scores)).rolling(smoothing, min_periods=smoothing).mean()
    return regularized_scores

def get_scores(args):
    if not os.path.isdir(args.experiments):
        logger.error("Dir at {} does not exists".format(args.experiments))
        exit(1)

    root, algortithms, _ = os.walk(args.experiments).__next__()
    score_data = {}
    for algortithm in algortithms:
        if not os.path.isdir(os.path.join(root, algortithm, args.environment)):
            logger.error("Algorithm folder {} does not have data for environment {}".format(algortithm, args.environment))
            continue

        for file in ['trainScores.t7', 'scores.t7', 'V.t7', 'losses.t7']:
            if not os.path.isfile(os.path.join(root, algortithm, args.environment, file)):
                logger.error("Environment {} does not have file {} for algorithm {}".format(args.environment, file, algortithm))
                continue

        score_data[algortithm] = {"trainScores": uniform_scores(os.path.join(root, algortithm, args.environment, 'trainScores.t7'), args.smoothing),
                                  "evalScores": uniform_scores(os.path.join(root, algortithm, args.environment, 'scores.t7')),
                                  "vEstimates": uniform_scores(os.path.join(root, algortithm, args.environment, 'V.t7')),
                                  "losses": uniform_scores(os.path.join(root, algortithm, args.environment, 'losses.t7')),
                                  "tdErrors": uniform_scores(os.path.join(root, algortithm, args.environment, 'TDErrors.t7'))}

    return score_data


def get_colors(color, choices):
    colors = list(Color(color[0]).range_to(Color(color[1]), len(choices)))
    return dict(zip(choices, ['rgb' + str(tuple(int(y * 255) for y in x.rgb)) for x in colors]))


def _add_stats_fig(color, fig, stats, title):
    rewards_graph = go.Scatter(
        x=list(range(len(stats["trainScores"]))),
        y=stats["trainScores"],
        line=dict(color=color, width=4, dash='solid'),
        mode='lines',
        legendgroup=title,
        name=title
    )
    length_graph = go.Scatter(
        x=list(range(len(stats["losses"]))),
        y=stats["losses"],
        line=dict(color=color, width=4, dash='solid'),
        mode='lines',
        legendgroup=title,
        name=title,
        showlegend=False
    )
    end_times_graph = go.Scatter(
        x=list(range(len(stats["vEstimates"]))),
        y=stats["vEstimates"],
        line=dict(color=color, width=4, dash='solid'),
        mode='lines',
        legendgroup=title,
        name=title,
        showlegend=False
    )
    fig.append_trace(rewards_graph, 1, 1)
    fig.append_trace(length_graph, 1, 2)
    fig.append_trace(end_times_graph, 1, 3)


def plotting(score_data, colors, args):
    logger.info("plotting the data")
    fig = tools.make_subplots(rows=1, cols=3, print_grid=False)
    if not os.path.exists(os.path.join(args.experiments, "plots")):
        os.makedirs(os.path.join(args.experiments, "plots"))

    for key, value in score_data.items():
        _add_stats_fig(colors[key], fig, value, key)
    fig["layout"].update(layout, title="Comparision-{}".format(args.environment))
    offline.plot(fig, filename=os.path.join(args.experiments, "plots", "Comparision-{}.html".format(args.environment)),
                 image_filename="Comparision-{}".format(args.environment),
                 image='png', image_width=1600, image_height=600)



if __name__ == '__main__':
    logger.info("Parsing the arguments")
    args = parse_args()

    score_data = get_scores(args)
    colors = get_colors(args.color, score_data.keys())
    print(colors)
    plotting(score_data=score_data, colors=colors, args=args)
