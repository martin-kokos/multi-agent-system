# Multi-agent-system

## Goal

The goal is to create an interesting simulation and encounter some of the way an simulation can explode and not go the way you expect. It uses elements of game thoery within an environment that simulates social interactions between agents.

### Checklist

- [x] environment
- [x] agents
    - [x] social interactions
    - [x] multiple abilitiess
    - [x] unique characteristics
    - [x] work towards common goal
- [x] use GT, DL, ML to design implement the system
- [x] scalable and efficient
- [x] docs
    - [x] brief summary
    - [x] description of system and assumptions
    - [x] agent architecture, abilities, behaviors
    - [x] experiment setup evaluation metrics used to asses the system's performance

## Solution

### Summary

TL;DR The Agents talk to each other about topics. When they agree, their confidence grows. Agents with highest confidence gain a restricted ammount of government seats for limited time. When there is an agreement between agents in government seat on a predetermined topic, the topic is marked as solved. Topics to be solved are picked randomly.

The simulation tries to capture the dynamics of the people being elected into a local government. People are simulated by agents and their goal is to have government which van *solve* public issues.
The goal is to introduce a method of abstraction of persons personality and social interactions based on it.

People are represented by *agents* which have unique trait of *confidence* and personal values and opinions towards public issues represented by *personality*. These prefereces form their communication. Two agents get to interact about one topic on an involutary basis as the environment decides to cross their _lives_. Their confidence increases each time their talk with another agent with a positive outcome - their communication was agreeable based on topic and their personality.
Agents with highest confidence are selected to gain a government position once there is a free slot in the government.
The purpose of the government is to solve current *issues* by means of having opnion that is similar-enough to the other gov members. These issues are part of the *environment*.

- *agent* is a model of a citizen
- *topic* (axis *t*) represents a 1-dimensional space of ordered *topics* in the [-1, 1] domain. The topics are assumed to have a 1-d solution space.
- *personality* is function f along an axis *t* representing a bi-polar opinion in the [-1, 1] domain.
- *public issue* is a single point along the axis of *topics*

### Usage

Install dependencies per pyproject.toml, eg. `poetry install`
- Run `python main.py` (with --plot to render animation frames as well)
- Use Jupyter Lab, `jupyter lab`
    - use the `sim_run.ipynb` [notebook](https://github.com/martin-kokos/multi-agent-system/blob/main/sim_run.ipynb) to run a simulation a see the outcome of the simulation.
    - use the `bayes_search.ipynb` [notebook](https://github.com/martin-kokos/multi-agent-system/blob/main/bayes_search.ipynb) to explore how different hyper-params affect the simulation characteristics

### Analysis

We see that some agents can lucky or unlucky with their personality in terms of growing their confidence.
The simulation is stable or stable in resonance for certain hyper-parameters, but for some combinations of hypermaters the simluation may quickly converge towards a locally stable, however uninteresting state, or explode into a chaotic state as there is no normalizating mechanism. The bayes_search notebook is provided to find hyper-parameters according to an objective function (defined in the notebook) which defines the desired properties of the simulation.

#### Illustration
The following animation presents the personality functions on topic/opinion space, through time.
- blue functions: the personalities of agents.
- red functions: the personalities of agents currently in government seat
- green vertical lines: current issues to be solved
- grey dots: interactions among agents taking place

![Peronalities changing in time](https://github.com/martin-kokos/multi-agent-system/raw/main/personalities_in_time.gif)

Notice that when the values of gov agents (red) at a topic (green) are clustered enough, the topic (green) is solved and disappears.

In order to draw conclusions from the simulation, we can take advantage of the simplicity of the model and observe events that would be otherwise be too rare to occur. Or we are able to run many simulations in order to do statistical analysis and draw conclusions from them, which would otherwise stay hidden in the noise of a single simulation.


#### Simulation tuning

In order to find simulation hyper-parameters which produce a simulation that best represents the real world that we are modeling, we can use the bayes_search notebook. The `objective` function returns the fitness of the properties of our simulation so an optimizer can search for the "fittest" combination of parameters. Multiple simulations with varying seeds are ran. The optimizer will produce a dict with params, which can be used in the sim_run notebook, in order to examine a simulation run.

### Improving the system

Through out the development process, I've come across many ideas to expand the simulation and could go on ad nauseum, so these are just some high-level ideas.
Feedback loops or normalization could be used to keep the simulation stable and true to design goals.
For the pupose of approximating an agent's personality, a 1-D polynomial was chosen because it illustrates the abstraction well. That could be improved, however the function chosen would be highly dependen't on the solution of the underlying problem of expressing the topic-space. In the case of many-deimensional topic space, the compexity of topics could be brought down with PCA.
The traits of the agents are generated uniformly, however it would be interesting how the system would behave if peoples traits would cluster around certain stereotypes, such as 16personalities.com, although I am not convinced personalities cluster.

## Performance

The source code was mostly optimized for readability and extendability.
The performance is about as good as it can be in python (Some easy performance could be gained by using Cython). Most time is spent on polyonmial fitting done by numpy.
Plot rendering when outputing plots also takes a lot of time (would use real timeseries stack for logging and visualization). 
It is possible to specify simulation hyper-parameters via command line arguments (see --help) in order to run independent simulations in parallel.
