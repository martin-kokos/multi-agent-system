import base64
import random
import logging
import json

import click
from collections import deque
from dataclasses import dataclass
from itertools import chain
from typing import Iterable, Iterator, Callable

import numpy as np
from tqdm.auto import tqdm

from metrics import Stats

log = logging.getLogger(__name__)


@dataclass(eq=False)
class Agent(object):
    """An agent representing a citizen.

    Params:
        env: instance of the Environment class for the agent to act in

        convo_disagreement: agent's disagreement tolerance. Higher number means agent
                            is more tolerant of different opinion and interaction is more likely to
                            result in positive feedback
        confidence_gain:    in case of positive/negative interaction, the afen'ts confidence will
                            be increased/decreased by this ammount
        evolve_rate:        a rate at which the agent adjusts it's personality
        poly_degree:        the degree of polynomial funciton which represents the agent's personality

    Attributes:
        Agent.confidence: Agent's confidence affects evolution of it's personality and also is a factor when
                          agent's are chose for government positions

    Methods:
        Agent.interact(topic)

    >>> agent = Agent(env=None, convo_disagreement=0.5, confidence_gain=0.2, evolve_rate=0.1, poly_degree=6)
    >>> response, feedback_cb = agent.interact(topic=0.3)
    >>> response
    1.3471731756666891
    >>> feedback_cb(topic=0.3, outcome=0)
    >>> agent.confidence
    0.25
    """

    env: "Environment"  # type lookahead
    convo_disagreement: float
    confidence_gain: float
    evolve_rate: float
    poly_degree: int

    entropy_bits: int = 64

    def __post_init__(self) -> None:
        self._feedback_received = True
        self.random = random.Random()
        if self.env:
            self.ident = self.env.random.getrandbits(128)
            self.random.seed(self.env.random.getrandbits(self.entropy_bits))
        else:
            self.ident = self.random.getrandbits(128)
            self.random.seed(0)
        self.name = base64.b32encode(self.random.randbytes(5)).decode() + f"_{self.ident}"
        self.personality = np.polynomial.Chebyshev([self.random.uniform(-1, 1) for _ in range(self.poly_degree)])
        self.experience = deque(maxlen=int(round(self.poly_degree * 2, 0)))
        self.strategy_weights = {
            "fuzz": 0.0,
            "smart": 1.0,
        }
        self.confidence = 0.05

    def __str__(self):
        return f"Agent:{self.name}"

    def __hash__(self):
        return hash(str(self.name))

    @property
    def _experienced_disagreement(self) -> float:
        """
        Return the ratio of negative interactions that are remembered.
        """
        return np.mean([e["outcome"] > self.convo_disagreement for e in self.experience])

    def _positive_feedback(self, gain: float) -> None:
        self.confidence = min([self.confidence + gain, 100])

    def _negative_feedback(self, gain: float) -> None:
        self.confidence = max([self.confidence - gain, 0.1])

    def interact(self, topic: float) -> tuple[float, Callable]:
        """
        Interaction given a topic produces an opinion which is compared with the opinion of the other party.

        The feedback callback function must be called, otherwise an assertion is raised.

        Returns:
            (opinion: float, feedback_callback: Callable)
        """
        assert self._feedback_received, "The feedback callback returned by Agent.interact() wasn't called."
        self._feedback_received = False

        return self.personality(topic), self._feedback

    def _feedback(self, topic: float, outcome: float) -> None:
        """
        A feedback function to receive and process interaction feedbackself.

        Must be called after every interaction.
        """
        self._feedback_received = True

        if abs(outcome) < self.convo_disagreement:
            self._positive_feedback(self.confidence_gain)
        else:
            self._negative_feedback(self.confidence_gain)

        self._evolve()

        self.experience.append(
            {
                "topic": topic,
                "opinion": self.personality(topic),
                "other_opinion": self.personality(topic) + outcome,
                "outcome": outcome,
            }
        )

    def _evolve(self) -> None:
        """
        Update agent's attributes and policies.
        """
        # being lost and overconfidence negatively impacts personality stability
        confidence_bowl = min([10, (self.confidence - 1) ** 2])

        evolve_rate = self.evolve_rate * confidence_bowl

        strategy = self.random.choices(list(self.strategy_weights.keys()), weights=self.strategy_weights.values(), k=1)[0]
        match strategy:
            case "fuzz":
                self._fuzz_evolve_strategy(evolve_rate)
            case "smart":
                self._smart_evolve_strategy(evolve_rate)

    def _fuzz_evolve_strategy(self, evolve_rate: float) -> None:
        """
        A randomness based mutation.
        """
        new_view = np.polynomial.Chebyshev([self.random.uniform(-1, 1) for _ in range(self.poly_degree)])
        self.personality = self.personality * (1 - evolve_rate) + new_view * evolve_rate

    def _smart_evolve_strategy(self, evolve_rate: float) -> None:
        """
        An experience based mutation.
        """
        if len(self.experience) == self.experience.maxlen:
            new_view = np.polynomial.Chebyshev.fit(
                x=[e["topic"] for e in self.experience],
                y=[e["other_opinion"] for e in self.experience],
                deg=self.poly_degree,
                domain=(-1, 1),
                rcond=0.5,
            )

            self.personality = self.personality * (1 - evolve_rate) + new_view * evolve_rate


@dataclass
class Environment(object):
    """An environment for agent based simulation representing the society and the government system.

    Params:
        seed: PRNG seed for reproducibility
        max_agents: size of agent population
        num_issues: number of concurrent issues
        gov_term: maximum number of ticks an agent can stay in government seat
        issue_solving_disagreement: measure of disagreemend allowed among agents in order to solve and dismiss an issue
    Params related to Agent instances
        agent_convo_disagreement: float = 0.5
        agent_confidence_gain: float = 0.1
        agent_evolve_rate: float = 0.5
        agent_poly_degree: int = 6

    Attributes:
        Encironment.stats: A statisctics tracking class

    Methods:
        Environment.run(ticks=1000)

    >>> env = Environment(seed=0, max_agents=10)
    >>> env.run(ticks=100)
    >>> env.time
    100
    """

    seed: int
    max_agents: int
    num_issues: int = 5
    gov_term: int = 50
    issue_solving_disagreement: float = 0.2

    agent_convo_disagreement: float = 0.5
    agent_confidence_gain: float = 0.1
    agent_evolve_rate: float = 0.5
    agent_poly_degree: int = 6

    def __post_init__(self) -> None:
        self.common_agents = list()
        self.random = random.Random()
        self.random.seed(self.seed)
        self.stats = Stats()
        self.time = 0
        self.max_gov = max([3, round(self.max_agents * 0.05)])
        self.convos_per_tick = round(2 * np.sqrt(self.max_agents))

        self.issues = list()
        self.num_solved_issues = 0
        self.gov = dict()

        for _ in range(self.max_agents):
            agent = Agent(
                env=self,
                convo_disagreement=self.agent_convo_disagreement,
                confidence_gain=self.agent_confidence_gain,
                evolve_rate=self.agent_evolve_rate,
                poly_degree=self.agent_poly_degree,
            )
            self.common_agents.append(agent)

        for _ in range(self.num_issues):
            self._spawn_issue()

    def _spawn_issue(self) -> None:
        self.issues.append(self.random.uniform(-1, 1))

    @property
    def gov_agents(self) -> Iterable:
        """
        Agents currently in government seat
        """
        return self.gov.keys()

    @property
    def iagents(self) -> Iterator:
        """Iterator of all agents in envirnment"""
        return chain.from_iterable([self.gov.keys(), self.common_agents])

    def agent_by_name(self, name: str) -> Agent:
        """
        Get an agent by it's name attribute
        """
        for agent in self.iagents:
            if agent.name == name:
                return agent
        else:
            raise KeyError

    def _interact_agents(self) -> None:
        left, right = self.random.sample(self.common_agents, k=2)

        topic = self.random.choice(self.issues) + self.random.uniform(-0.1, 0.1)

        left_opinion, left_feedback_cb = left.interact(topic)
        right_opinion, right_feedback_cb = right.interact(topic)

        disagreement = left_opinion - right_opinion

        left_feedback_cb(topic=topic, outcome=-disagreement)
        right_feedback_cb(topic=topic, outcome=disagreement)

        self.stats.event(
            "conversation",
            time=self.time,
            tags={"disagreement": disagreement, "topic": topic, "mid_opinion": np.mean([left_opinion, right_opinion])},
        )

    def _promote_agent(self, agent) -> None:
        self.common_agents.remove(agent)
        self.gov[agent] = self.gov_term

    def _demote_agent(self, agent) -> None:
        del self.gov[agent]
        self.common_agents.append(agent)

    def _elect_gov(self) -> None:
        if len(self.gov) >= self.max_gov:
            log.debug("Gov is full. No elections.")
            return None

        nominees_ranked = sorted(self.common_agents, key=lambda a: a.confidence, reverse=True)
        winner = nominees_ranked[0]
        self._promote_agent(winner)
        self.stats.log(winner.name, f"Agent {winner.name} has been elected [{self.time}]")

    def _cleanup_gov(self) -> None:
        for expired_gov_agent in [a for a, term_left in self.gov.items() if term_left < 1]:
            self._demote_agent(expired_gov_agent)
            self.stats.log(expired_gov_agent.name, f"Agent {expired_gov_agent.name} ended gov term [{self.time}]")

    def _gov_run(self) -> None:
        """
        Do government related housekeeping
        """
        if len(self.gov) >= self.max_gov:
            self._solve_issue(agents=self.gov.keys())

        # decrement term left
        for a in self.gov.keys():
            self.gov[a] = self.gov[a] - 1

        # wear down confidenece
        for a in self.gov.keys():
            a._negative_feedback(gain=a.confidence / self.gov_term)

        self._cleanup_gov()
        self._elect_gov()

    def _solve_issue(self, agents) -> None:
        for issue in self.issues:
            issue_outcomes = [agent.personality(issue) for agent in agents]
            disagreement = np.std(issue_outcomes)
            if disagreement < self.issue_solving_disagreement:
                self.num_solved_issues += 1
                self.issues.remove(issue)
                log.info(f"Issue {issue:.2} solved [{disagreement:.2}].")

                for a in self.gov.keys():
                    a._positive_feedback(gain=self.agent_confidence_gain)

        if len(self.issues) < self.num_issues:
            self._spawn_issue()

    def _record_metrics(self) -> None:
        """
        Record the current metrics in a metrics.Stats class


        Metrics:
            issues.solved:
            gov.num_agents:
            agents.mean_confidence:
            agents.mean_poly_complexity: Mean sum of absolute values of coeficients of polynomes in personalities
            issues.gov_agreement: Gov agreement on current issues
        """
        self.stats.gauge("time", self.time)

        for a in self.iagents:
            self.stats.gauge(f"agent:{a.name}.confidence", a.confidence)

        self.stats.gauge("issues.solved", self.num_solved_issues)
        self.stats.gauge("gov.num_agents", len(self.gov))
        self.stats.gauge("agents.mean_confidence", float(np.mean([a.confidence for a in self.iagents])))
        self.stats.gauge(
            "agents.mean_poly_complexity", np.mean(list(chain(abs(c) for c in a.personality.coef for a in self.iagents)))
        )
        self.stats.gauge(
            "issues.gov_agreement",
            1 / (sum(np.std([gov_agent.personality(issue) for gov_agent in self.gov_agents]) for issue in self.issues) + 0.0001),
        )

    def _tick(self) -> None:
        for _ in range(self.convos_per_tick):
            self._interact_agents()
        self._gov_run()
        self.time += 1

    def run(self, ticks=1000) -> None:
        run_msg = f"Running {self.max_agents} agents, gov size {self.max_gov}, {self.convos_per_tick} itx/t, {ticks} ticks"

        for _ in tqdm(range(ticks), desc=run_msg, leave=False):
            self._tick()

            self._record_metrics()


@click.command()
@click.option("--max_agents", default=200, help="Number of agents")
@click.option("--agent_confidence_gain", default=0.3, help="Confidence gein when agent has positive interaction")
@click.option("--agent_convo_disagreement", default=0.5, help="Threshold of agent having a negative interaction")
@click.option("--agent_evolve_rate", default=0.4, help="Multiplier how fast agent adjust itself")
@click.option("--agent_poly_degree", default=4, help="Degree of olyynome of agent's personality")
@click.option("--gov_term", default=100, help="How many ticks agent stays in government seat")
@click.option("--issue_solving_disagreement", default=0.15, help="Disagreement threshold for solving issues")
@click.option("--num_issues", default=10, help="Number of concurrent issues")
@click.option("--ticks", default=1000, help="Number of ticks to run the simulation")
@click.option("--seed", default=0, help="Random seed")
@click.option("--plot", is_flag=True, help="Create an animation of personalities")
@click.option("--verbose", default=False, help="Verbosity")
def run_sim(**kwargs):
    ticks = kwargs.pop("ticks")
    plot = kwargs.pop("plot")
    verbose = kwargs.pop("verbose")

    if verbose:
        logging.basicConfig(level=logging.INFO)

    kwargs = {
     'seed': 0,
     'max_agents': 200,

    'agent_confidence_gain': 0.1,
     'agent_convo_disagreement': 0.25,
     'agent_evolve_rate': 0.5,
     'agent_poly_degree': 5,
     'gov_term': 50,
     'issue_solving_disagreement': 0.05,
     'num_issues': 5}

    env = Environment(**kwargs)

    if not plot:
        env.run(ticks=ticks)

    else:
        for i in tqdm(range(int(ticks / 10))):
            step = 10
            env.stats.save_plot(env, frame_num=i * step)
            env.run(ticks=step)

        import subprocess
        subprocess.run('./mk_video.sh', shell=True)
        print('Saved personalities_in_time.gif')
        print()

    summary = env.stats.summary_metrics()
    print(json.dumps(summary, sort_keys=True, indent=4))


if __name__ == "__main__":
    run_sim()
