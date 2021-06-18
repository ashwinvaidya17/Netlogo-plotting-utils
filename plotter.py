import hashlib
import math
import os
import sys

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tqdm
from numpy.core.fromnumeric import sort

csv_file = "all_runs.csv"


data = pd.read_csv(csv_file)
data = data.transpose()
data.columns = data.iloc[0]
data = data[1:]
data.head()


data = data.fillna(method="ffill")
data.head()

# change columns according to the experiment
columns = [
    "flock-detection-range",
    "error-margin",
    "noise-stddev",
    "max-separate-turn",
    "acc-dist-threshold",
    "acceleration",
    "population",
    "toggle-n-topo",
    "error-in-flight-direction",
    "update-freq",
    "front-partnert-dist-check",
    "max-speed",
    "speed-stddev",
    "separation-timeout",
    "max-vision",
    "max-align-turn",
    "n-topo-neighbours",
    "min-speed",
    "pair-bonding",
    "vision-reduction-factor",
    "partner-check-fov",
    "max-cohere-turn",
    "back-partnert-dist-check",
    "dec-dist-threshold",
    "minimum-separation",
    "speed",
    "FOV",
    "deceleration",
    "[reporter]",
    "[final]",
]


data = data[columns]


max_runs = int(data.index[-1].split(".")[0])


values = {}
for column in data.columns:
    values[column] = set(data[column])


reporters = set(data["[reporter]"])


constants = {}
variables = {}
for column in values.keys():
    if len(values[column]) == 1:
        constants[column] = values[column]
    else:
        variables[column] = values[column]
try:
    variables.pop("[reporter]")
    variables.pop("[final]")
except KeyError:
    pass  # if key does not exist


def plotter(const_list, remaining_list):

    # recursively calls plotter to hold different values constant
    if len(remaining_list) > 2:
        for i in range(len(remaining_list)):
            for const in remaining_list[i][1]:
                curr_const = const_list + [(remaining_list[i][0], const)]
                n_remaining = remaining_list[:i] + remaining_list[i + 1 :]
                plotter(curr_const, n_remaining)

    elif len(remaining_list) == 2:
        x_vals = remaining_list[0][1]
        # second metric is used to plot other lines used for comparison
        second_metric = remaining_list[1][1]
        second_metric_name = remaining_list[1][0]
        collated_reporters = []
        flattened_name = ""
        for i in const_list:
            k, v = i
            flattened_name += f"{k}_{list(v)[0]}"
        for s_metric_val in second_metric:
            reporter_values = {}
            for k in reporters:
                reporter_values[k] = []
            df = data
            # go over all the values held constant for this plot
            for i in const_list:
                k, v = i
                df = df.loc[df[k] == list(v)[0]]

            df = df.loc[df[second_metric_name] == s_metric_val]
            for val in x_vals:
                curr_df = df.loc[df[remaining_list[0][0]] == val]
                for k in reporters:
                    reporter_values[k].append(curr_df.loc[curr_df["[reporter]"] == k]["[final]"].item())
                    # add values to respective reporters
            collated_reporters.append(reporter_values)

        # go over each reporter and make a plot for it
        for k in reporters:
            plt.clf()
            for i, entry in enumerate(collated_reporters):
                try:
                    plt.plot(
                        sort([float(i) for i in list(x_vals)]),
                        [float(i) for i in entry[k]],
                        label=f"{second_metric_name}: {list(second_metric)[i]}",
                    )
                except ValueError:
                    return  # axis selected is boolean

            plt.legend()
            plt.ylabel(k)
            plt.xlabel(remaining_list[0][0])
            f_name = hashlib.md5(f"{flattened_name}_{k}".encode()).hexdigest()
            with open(os.path.join("plots/", f"{f_name}.txt"), "w") as f:
                f.write(f"{flattened_name}_{k}")
            plt.savefig(os.path.join("plots", f"{f_name}.png"))
    else:  # only one variable
        x_vals = remaining_list[0][1]
        collated_reporters = []
        flattened_name = ""
        for i in const_list:
            k, v = i
            flattened_name += f"/{k}_{list(v)[0]}"
        reporter_values = {}
        for k in reporters:
            reporter_values[k] = []
        df = data
        # go over all the values held constant for this plot
        for i in const_list:
            k, v = i
            df = df.loc[df[k] == list(v)[0]]

        for val in x_vals:
            curr_df = df.loc[df[remaining_list[0][0]] == val]
            for k in reporters:
                reporter_values[k].append(curr_df.loc[curr_df["[reporter]"] == k]["[final]"].item())
                # add values to respective reporters
        collated_reporters.append(reporter_values)

    # go over each reporter and make a plot for it
    for k in reporters:
        plt.clf()
        for i, entry in enumerate(collated_reporters):
            try:
                plt.plot(
                    sort([float(i) for i in list(x_vals)]), [float(i) for i in entry[k]],
                )
            except ValueError:
                return  # axis selected is boolean

        plt.ylabel(k)
        plt.xlabel(remaining_list[0][0])
        f_name = hashlib.md5(f"{flattened_name}_{k}".encode()).hexdigest()
        with open(os.path.join("plots/", f"{f_name}.txt"), "w") as f:
            f.write(f"{flattened_name}_{k}")
        plt.savefig(os.path.join("plots", f"{f_name}.png"))


plotter(list(constants.items()), list(variables.items()))

