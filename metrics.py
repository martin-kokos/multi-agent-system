from collections import defaultdict
from itertools import cycle
from typing import Iterable, Sequence

import bokeh.plotting as plt
import bokeh.palettes as palettes
from bokeh.models import Span

from matplotlib.patches import Patch
import matplotlib.pyplot as mp_plt


import numpy as np

from logging import getLogger

log = getLogger(__name__)


class Stats(object):
    """A simple statsd-like stats collector."""

    X_space = np.linspace(-1, 1, 21)

    def __init__(self) -> None:
        self.store = defaultdict(list)
        self.logs = defaultdict(list)
        self.events = list()
        self.img_frame = 0

    def gauge(self, tag: str, value: float) -> None:
        self.store[tag].append(value)

    def event(self, event: str, time: int, tags: dict) -> None:
        self.events.append({"event_name": event, "time": time, **tags})

    def get_events(self, before: int | None = None, after: int | None = None) -> list:
        return [
            event
            for event in self.events
            if (before is None or event["time"] < before and after is None or event["time"] >= after)
        ]

    def summary_metrics(self) -> dict:
        def var_from_first(series: Sequence) -> float:
            """
            Variance but based on first element instead of meanself.

            Expresses stability compared to initial state.
            """
            first = series[0]
            se_from_first = [(first - e) ** 2 for e in series]
            return sum(se_from_first) / len(series)

        elapsed_time = self.store["time"][-1]
        issues_solved = self.store["issues.solved"][-1]
        issues_solved_rate = self.store["issues.solved"][-1] / elapsed_time
        issues_solved_auc = sum(self.store["issues.solved"])
        poly_complexity_var = var_from_first(self.store["agents.mean_poly_complexity"])
        confidence_var = np.var(self.store["agents.mean_confidence"])

        return {
            "elapsed_time": elapsed_time,
            "issues_solved": issues_solved,
            "issues_solved_rate": issues_solved_rate,
            "issues_solved_auc": issues_solved_auc,
            "poly_complexity_var": poly_complexity_var,
            "confidence_var": confidence_var,
        }

    def plot(self, tag: str, figure=None, **kwargs):
        p = figure or plt.figure(
            title=f"{tag}* line plots (resampled)",
            x_axis_label="t",
            y_axis_label="y",
            sizing_mode="stretch_width",
            max_width=800,
            **kwargs,
        )

        matching_keys = [k for k in self.store.keys() if k.startswith(tag)]

        palette = cycle(palettes.linear_palette(palettes.Turbo256, min(256, len(matching_keys))))

        for t, color in zip(
            matching_keys,
            [next(palette) for _ in matching_keys],
        ):
            series = self.store[t]
            series_length = len(series)
            # simplify busy plot
            series = np.interp(
                x=np.linspace(0, series_length, round(p.width / np.log(10 * len(matching_keys) + 1))),
                xp=range(series_length),
                fp=series,
            )
            p.xaxis.major_label_text_font_size = "0pt"  # labels don't match after resampling

            p.line(list(range(len(series))), series, line_width=1, legend_label=t, color=color)
        p.legend.visible = False
        plt.show(p)
        return p

    def log(self, key: str, msg: str) -> None:
        log.debug(f"{str}: {msg}")
        self.logs[key].append(msg)

    def get_log(self, key: str) -> None:
        return self.logs[key]

    @staticmethod
    def plot_personalities(env):
        X = np.linspace(-1, 1, 21)
        p = plt.figure(title="Population personalities", x_axis_label="x", y_axis_label="y", x_range=(-1, 1), y_range=(-2, 2))

        def plot(f, **kwargs):
            y = [f(x) for x in X]
            p.line(X, y, **kwargs)

        for agent in env.common_agents:
            plot(
                agent.personality,
                line_width=1,
                alpha=0.1,
            )

        for agent in env.gov_agents:
            plot(agent.personality, line_width=2, alpha=1, color="red")

        spans = []
        for issue in env.issues:
            spans.append(Span(location=issue, dimension="height", line_color="green"))

        p.renderers.extend(spans)
        return p

    def plot_personalities_matplotlib(self, env, frame_num):
        fig = mp_plt.Figure()
        ax = fig.add_subplot()

        mp_plt.rc("figure", figsize=(10, 10), dpi=72)
        mp_plt.grid(True)
        mp_plt.xlim([-1, 1])
        mp_plt.ylim([-2, 2])
        mp_plt.gca().set_xlabel("topic")
        mp_plt.gca().set_ylabel("opinion")
        mp_plt.title(f"Current personalities: frame {frame_num}")

        def plot(f, **kwargs):
            y = [f(x) for x in self.X_space]
            mp_plt.plot(self.X_space, y, **kwargs)

        for agent in env.common_agents:
            plot(agent.personality, alpha=0.1, linewidth=1, color="blue", label="agent")

        for agent in env.gov_agents:
            plot(agent.personality, linewidth=2, color="red", label="gov agent")

        for issue in env.issues:
            mp_plt.axvline(x=issue, color="green", label="issue")

        for interaction in env.stats.get_events(before=frame_num, after=max([0, frame_num - 10])):
            mp_plt.scatter(
                x=interaction["topic"],
                y=interaction["mid_opinion"],
                color="black",
                marker=".",
                alpha=0.5,
            )

        fig.legend(
            handles=[
                Patch(facecolor="red", edgecolor="k", label="gov agent"),
                Patch(facecolor="blue", edgecolor="k", label="agent"),
                Patch(facecolor="grey", edgecolor="k", label="agent talk"),
                Patch(facecolor="green", edgecolor="k", label="topic"),
            ],
            loc="outside lower center",
        )
        mp_plt.savefig(f"img/personalities_{self.img_frame:05}.png")
        mp_plt.clf()
        self.img_frame += 1

    def save_plot(self, env, frame_num):
        # mp_plt.style.use('fast')
        self.plot_personalities_matplotlib(env, frame_num)
