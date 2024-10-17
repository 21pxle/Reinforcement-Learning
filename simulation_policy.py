import abc
from collections import namedtuple


class State:
    def __init__(self, episode_num: int, truncated_times: int, terminated_times: int):
        self.episode_num = episode_num
        self.truncated_times = truncated_times
        self.terminated_times = terminated_times


class SimulationPolicy:

    def __init__(self):
        pass

    @abc.abstractmethod
    def can_continue(self, state: State):
        pass


class SimulateUntilTerminated(SimulationPolicy):

    def __init__(self, terminated_times: int = 1):
        super().__init__()
        self.terminated_times = terminated_times

    def can_continue(self, state: State):
        return state.terminated_times < self.terminated_times


class SimulateUntilTruncated(SimulationPolicy):

    def __init__(self, truncated_times: int = 1):
        super().__init__()
        self.truncated_times = truncated_times

    def can_continue(self, state: State):
        return state.truncated_times < self.truncated_times


class SimulateNumberOfEpisodes(SimulationPolicy):

    def __init__(self, n_episodes: int):
        super().__init__()
        self.n_episodes = n_episodes

    def can_continue(self, state: State):
        return state.episode_num < self.n_episodes


class PolicyAnd(SimulationPolicy):
    def __init__(self, policy1: SimulationPolicy, policy2: SimulationPolicy):
        super().__init__()
        self.policy1 = policy1
        self.policy2 = policy2

    def can_continue(self, state: State):
        return self.policy1.can_continue(state) and self.policy2.can_continue(state)


class PolicyOr(SimulationPolicy):
    def __init__(self, policy1: SimulationPolicy, policy2: SimulationPolicy):
        super().__init__()
        self.policy1 = policy1
        self.policy2 = policy2

    def can_continue(self, state: State):
        return self.policy1.can_continue(state) or self.policy2.can_continue(state)
