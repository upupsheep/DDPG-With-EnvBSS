import gym
import numpy as np
import os.path
from gym import error, spaces, warnings
from gym.utils import seeding
"""
### Nouns: ###

capacity:
allocation: 要分配給zone s的車車數
flow: timestep t時從zone i移動到zone j的車車數輛?
demand: 顧客會從zone x騎到zone y
    - demand_1d: zone_x在timestep t時總共移出的車車數

### Variables: ###

self.nzones: 95 *
self.timesteps: 12 *
self.nbikes: 760 -> 總共多少輛車車 *
self.max_demand: 100 -> timestep t時在zone s的最大demand?

self.capacities: 每個zone可以容納的車車數 (一開始=self.__cp)
    - 1d array: shape = nzones
    - paper說"using 10 bike trailers, each having capacity of 5
self.__cp: 某個capacity?

self.starting_allocation: 一開始的allocation? (一開始=self.__ds)
self.__ds: 在timestep t時，station s裡擁有的車車數
    - 2d array: self.__ds[t][s]
    - self.__ds[t][s] -= (yp - yn)
    - yp: action預計移出, yn: action預計移入

self.demand_data: 需求?
    - self.demand_data[scenario] = flow

self.__dis: zone x跟zone y之間的距離
    - 2d array: self.__dis[x][y]
self.__mindis: zone x跟zone y之間的最短距離?
    - 2d array

self.__fl: flow? -> demand flow
self.__xfl: 3d array, timestep i時zone k 真正有移到zone j的flow?
self.__tfl1: 2d array, timestep t時從zone s到其他所有zone的flow的總和

### Observation: ###
1. the distribution of bikes at the end of the frame
2. the cumulative demand per zone during the 30-min frame
3. the time of the day (6hr, 12*t)

"""


class BSSEnv(gym.Env):
    def __init__(self,
                 nzones=95,
                 ntimesteps=12,
                 data_dir=None,
                 data_set_name='actual_data_art',
                 scenarios=list(range(1, 21))):
        super().__init__()
        self.nzones = nzones
        self.ntimesteps = ntimesteps
        self.scenarios = scenarios
        self.data_set_name = data_set_name
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "default_data")
        self.data_dir = data_dir
        self.__read_data()
        self.capacities = np.array(self.__cp)
        self.starting_allocation = np.array(self.__ds)
        self.max_demand = 100
        self.metadata = {
            'render.modes': [],
            'nzones': self.nzones,
            'ntimesteps': self.ntimesteps,
            'nbikes': self.nbikes,
            'capacities': self.capacities,
            'data_dir': self.data_dir,
            'scenarios': self.scenarios
        }
        self.observation_space = spaces.Box(
            low=np.array([0] * (2 * self.nzones + 1)),
            high=np.array([self.max_demand] * self.nzones +
                          list(self.capacities) + [self.ntimesteps]),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.zeros([self.nzones]),
            high=self.capacities,
            dtype=np.float32)
        self._scenario = 20
        self.seed(None)

    def __read_data(self):
        self.__read_capacity_and_starting_allocation(
            os.path.join(self.data_dir, "demand_bound_artificial_60.txt"))
        self.__read_zone_distances(
            os.path.join(self.data_dir, "RawData", "distance_zone.txt"))
        self.__read_demand_data(
            self.scenarios,
            os.path.join(self.data_dir, "DemandScenarios", self.data_set_name,
                         "DemandScenarios1", "demand_scenario_{scenario}.txt"))
        """
        self.__read_capacity_and_starting_allocation(
            os.path.join(
                os.path.dirname(__file__), "../../../demand_bound.txt"))
        self.__read_zone_distances(
            os.path.join(
                os.path.dirname(__file__), "../../../distance_zone.txt"))
        self.__read_demand_data(
            self.scenarios,
            os.path.join(
                os.path.dirname(__file__),
                "../../../demand_scenario/demand_scenario_{scenario}.txt"))
        """

    def __read_capacity_and_starting_allocation(self, filename):
        f = open(filename)
        line = f.readline()
        self.nzones = int(line)

        self.__cp = [0 for k in range(self.nzones)]
        self.__ds = [[0.0 for k in range(self.nzones)]
                     for j in range(self.ntimesteps + 1)
                     ]  # Distribution is zones
        # f = open(filename)
        # line = f.readline()

        line = f.readline()
        line = line.strip(" \n")
        line = line.split(" ")
        for s in range(self.nzones):
            self.__cp[s] = int(line[s])

        line = f.readline()
        line = line.strip(" \n")
        line = line.split(" ")
        self.nbikes = 0
        for s in range(0, self.nzones):
            self.__ds[0][s] = int(line[s])
            self.nbikes = self.nbikes + self.__ds[0][s]

        f.close()

    def __read_demand_data(self, scenarios, filename_unformatted):
        self.demand_data = {}
        for scenario in scenarios:
            flow = [[[0.0 for k in range(self.nzones)]
                     for j in range(self.nzones)]
                    for i in range(self.ntimesteps)]  # Known Flow
            f2 = open(filename_unformatted.format(scenario=scenario))
            for i in range(self.ntimesteps):
                for j in range(self.nzones):
                    line = f2.readline()
                    line = line.strip(" \r\n")
                    line = line.split(" ")
                    for k in range(self.nzones):
                        flow[i][j][k] = float(line[k])
            f2.close()
            self.demand_data[scenario] = flow

    def __read_zone_distances(self, filename):
        self.__dis = [[0.0 for k in range(self.nzones)]
                      for i in range(self.nzones)]
        f2 = open(filename)
        line = f2.readline()
        ma = 0
        T = 0
        for T in range(self.nzones):
            line = line.strip(' \r\n ')
            line = line.split(" ")
            for i in range(self.nzones):
                self.__dis[T][i] = float(line[i])  # /10000.0
                if (self.__dis[T][i] > ma):
                    ma = self.__dis[T][i]
            line = f2.readline()
        f2.close()

        for i in range(self.nzones):
            self.__dis[i][i] = 0

        self.__mindis = [[-1 for k in range(self.nzones)]
                         for i in range(self.nzones)]
        for i in range(self.nzones):
            sortindex = sorted(
                range(len(self.__dis[i])), key=lambda k: self.__dis[i][k])
            for j in range(self.nzones):
                self.__mindis[i][j] = sortindex[j]

    def seed(self, seed=None):
        if seed is None:
            seed = seeding.create_seed(max_bytes=4)
        self.__nprandom = np.random.RandomState(seed)
        return [seed]

    def _get_observation(self):
        if self.__t == 0:
            demand_2d = np.zeros(shape=[self.nzones, self.nzones])
        else:
            demand_2d = np.array(self.__fl[self.__t - 1])
        assert list(demand_2d.shape) == [self.nzones, self.nzones]
        demand_1d = np.sum(demand_2d, axis=1)
        alloc = np.array(self.__ds[self.__t])
        obs = np.concatenate([demand_1d, alloc, [self.__t]])
        assert list(obs.shape) == list(self.observation_space.shape)
        return obs

    def __reset_allocation(self):
        self.__ds = list(self.starting_allocation)

    def __reset_flow(self, scenario):
        self.__fl = self.demand_data[scenario]
        self.__xfl = [[[0.0 for k in range(self.nzones)]
                       for j in range(self.nzones)]
                      for i in range(self.ntimesteps)]  # Actual computed Flow
        self.__tfl1 = [[0.0 for k in range(self.nzones)]
                       for j in range(self.ntimesteps)]

        for t in range(0, self.ntimesteps):
            for s in range(0, self.nzones):
                for s1 in range(0, self.nzones):
                    self.__tfl1[t][s] = self.__tfl1[t][s] + self.__fl[t][s][s1]

    def reset(self):
        # pick up a day at random
        self._scenario = self.scenarios[self.__nprandom.randint(
            len(self.scenarios))]
        # self._scenario = self._scenario + 1
        # print("demand scenario is:", self._scenario)
        self.__reset_allocation()
        self.__reset_flow(self._scenario)

        self.__yp = [[0.0 for k in range(self.nzones)]
                     for j in range(self.ntimesteps)]
        self.__yn = [[0.0 for k in range(self.nzones)]
                     for j in range(self.ntimesteps)]
        self.__t = 0

        return self._get_observation()

    def __set_yp_yn_from_action(self, action):
        if action is None:
            warnings.warn(
                "no action was provided. taking default action of not changing allocation"
            )
        else:
            action = np.array(action)
            if not (hasattr(action, 'shape')
                    and list(action.shape) == list(self.action_space.shape)):
                raise error.InvalidAction(
                    'action shape must be as per env.action_space.shape. Provided action was {0}'.
                    format(action))
            if np.round(np.sum(action)) != self.nbikes:
                raise error.InvalidAction(
                    'Dimensions of action must sum upto env.metadata["nbikes"]. Provided action was {0} with sum {1}'.
                    format(action, sum(action)))
            if np.any(action < -1e-6):
                raise error.InvalidAction(
                    'Each dimension of action must be positive. Provided action was {0}'.
                    format(action))
            if np.any(action > self.capacities + 1e-6):
                raise error.InvalidAction(
                    'Individual dimensions of action must be less than respective dimentions of env.metadata["capacities"]. Provided action was {0}'.
                    format(self.capacities - action))
            # print("action: ", action)
            # print("current_alloc", self.__ds[self.__t])
            alloc_diff = action - np.array(self.__ds[self.__t])
            yn = alloc_diff * (alloc_diff > 0)
            yp = -alloc_diff * (alloc_diff < 0)
            self.__yp[self.__t] = list(yp)
            self.__yn[self.__t] = list(yn)

    def __calculate_lost_demand_new_allocation(self):
        full_lost = 0.0
        iteration = self.__t
        moving_cost = (sum(self.__yp[iteration])+sum(self.__yn[iteration]))/2
        assert abs(sum(self.__yp[iteration]) - sum(self.__yn[iteration])
                   ) < 1e-6, "sum(yp)={0}\nsum(yn)={1}\nyp={2}\nyn={3}".format(
                       sum(self.__yp[iteration]), sum(self.__yn[iteration]),
                       self.__yp[iteration], self.__yn[iteration])
        assert np.all(np.array(self.__yp[iteration]) >= -0.0)
        assert np.all(np.array(self.__yn[iteration]) >= -0.0)
        before_reallocation = sum(self.__ds[iteration])
        # print("Sum before reallocation:", before_reallocation)
        for s in range(self.nzones):
            # and ((yn[iteration][s]-yp[iteration][s])<=cp[s]-ds[iteration][s])):
            if ((self.__ds[iteration][s] >=
                 (self.__yp[iteration][s] - self.__yn[iteration][s]))):
                self.__ds[iteration][s] = self.__ds[iteration][s] - \
                    (self.__yp[iteration][s] - self.__yn[iteration][s])
            # elif((self.__yn[iteration][s] - self.__yp[iteration][s]) > self.__cp[s] - self.__ds[iteration][s]):
            #     self.__ds[iteration][s] = self.__cp[s]
            else:
                self.__ds[iteration][s] = 0.0

        for s in range(self.nzones):
            for s1 in range(self.nzones):
                # if(self.__tfl1[iteration][s] <= self.__ds[iteration][s]):
                #     self.__xfl[iteration][k][s][s1] = self.__fl[iteration][k][s][s1]
                # else:
                if (self.__tfl1[iteration][s] > 0):
                    self.__xfl[iteration][s][s1] = min(
                        self.__ds[iteration][s], sum(
                            self.__fl[iteration][s])) * (
                                self.__fl[iteration][s][s1] /
                                (self.__tfl1[iteration][s] * 1.0))

        for i in range(self.nzones):
            self.__ds[iteration + 1][i] = self.__ds[iteration][i] - \
                min(self.__ds[iteration][i], sum(self.__fl[iteration][i]))

        for z in range(self.nzones):
            for z1 in range(self.nzones):
                if (sum(self.__fl[iteration][z1]) > 0):
                    # (1.0*min(ds[iteration][z1],sum(fl[iteration][z1]))*fl[timstep][z1][z])/sum(fl[iteration][z1])
                    self.__ds[iteration + 1][z] = self.__ds[iteration + 1][z] + \
                        self.__xfl[iteration][z1][z]

        flag = 0

        after_reallocation = sum(self.__ds[iteration + 1])
        # print("Sum after reallocation:", sum(self.__ds[iteration + 1]))
        assert abs(
            after_reallocation - before_reallocation
        ) < 1e-6, "This is where the bug is. sum before reallocation={0}. sum after reallocation={1}\nallocation_before={2}\nallocation_after={3}".format(
            before_reallocation, after_reallocation, self.__ds[iteration],
            self.__ds[iteration + 1])

        while (flag == 0):
            for s in range(self.nzones):
                if (self.__ds[iteration + 1][s] > self.__cp[s]):
                    # print("readjusting for zone", s)
                    for s1 in self.__mindis[s]:
                        if ((self.__ds[iteration + 1][s] - self.__cp[s]) <=
                                (self.__cp[s1] - self.__ds[iteration + 1][s1])):
                            self.__ds[iteration + 1][
                                s1] = self.__ds[iteration +
                                                1][s1] - self.__cp[s] + self.__ds[iteration
                                                                                  +
                                                                                  1][s]
                            full_lost += self.__ds[iteration +
                                                   1][s] - self.__cp[s]
                            self.__ds[iteration + 1][s] = self.__cp[s]
                            break
                        elif (
                            ((self.__cp[s1] - self.__ds[iteration + 1][s1]) >
                             0) and
                            ((self.__ds[iteration + 1][s] - self.__cp[s]) >
                             (self.__cp[s1] - self.__ds[iteration + 1][s1]))):
                            self.__ds[iteration + 1][s] = self.__ds[iteration + 1][s] - \
                                (self.__cp[s1] - self.__ds[iteration + 1][s1])
                            full_lost += self.__cp[s1] - \
                                self.__ds[iteration + 1][s1]
                            self.__ds[iteration + 1][s1] = self.__cp[s1]

            after_readjustment_sum = sum(self.__ds[iteration + 1])
            assert abs(
                after_reallocation - after_readjustment_sum
            ) < 1e-6, "This is where the bug is. starting_sum={0}. After readjustment, sum={1}".format(
                after_reallocation, after_readjustment_sum)

            flag = 1
            for s in range(self.nzones):
                assert self.__ds[iteration +
                                 1][s] <= self.__cp[s], "I am stuck. Something is wrong. Readjustment should have finished in one pass"

        lost_call = 0
        revenue = 0
        for s in range(self.nzones):
            for s1 in range(self.nzones):
                revenue += self.__xfl[iteration][s][s1]
                lost_call += self.__fl[iteration][s][s1] - \
                    self.__xfl[iteration][s][s1]

        return lost_call, full_lost, moving_cost, revenue

    def step(self, action):
        # modify yp and yn here according to action
        self.__set_yp_yn_from_action(action)
        lost_demand, full_lost_demand, moving_cost_demand, revenue = self.__calculate_lost_demand_new_allocation(
        )
        r = -(lost_demand + full_lost_demand + 2*moving_cost_demand)
        self.__t += 1
        done = self.__t >= self.ntimesteps
        return self._get_observation(), r, done, {
            "lost_demand_pickup": lost_demand,
            "lost_demand_dropoff": full_lost_demand,
            "revenue": revenue,
            "scenario": self._scenario
        }

    def render(self, mode='human', close=False):
        if not close:
            raise NotImplementedError(
                "This environment does not support rendering")
