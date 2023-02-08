import os
import itertools
import gym
import numpy as np
import random
import json
from Environment.Technician import Technician
from Environment.Machine import Machine
import pygame
import matplotlib.pyplot as plt
import io
from pygame import gfxdraw
#os.environ["SDL_VIDEODRIVER"] = "dummy"

class FactoryEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 2}
    def __init__(self, path_config, config):
        f = open(path_config)
        x = json.load(f)
        fail_dist = [tuple(i) for i in x[config]["fail_dist"]]
        machines = [Machine(idx=i, failure_dist=fail_dist) for i in range(x[config]["number_machines"])]
        technicians = [Technician(t, i) for i, t in enumerate(x[config]["technicians"])]
        multiplier_length_episode = x[config]["multiplier_length_episode"]

        np.random.seed(0)
        random.seed(0)
        self.movetech = 0
        self.current_step = 0
        self.machines = machines
        self.technicians = technicians
        self.num_machines = len(machines)
        self.num_technicians = len(technicians)
        self.num_components = len(machines[0].life_components)
        self.max_episode_length = 0
        self.screen = None
        for d in self.machines[0].failure_dist:
            max_value = d.quantile(1.0 - np.finfo(float).eps)
            if self.max_episode_length < max_value:
                self.max_episode_length = max_value
        self.max_episode_length = np.ceil((multiplier_length_episode * self.max_episode_length) + 1)
        maxRepairTime = 0
        for t in self.technicians:
            for r in t.mt2r:
                maxRepairTime = r if r > maxRepairTime else maxRepairTime

        lc = [self.max_episode_length] * self.num_components  # Lc
        sc = [2] * self.num_components  # Sc
        rm = [maxRepairTime + 1]  # Rm
        n_tech = [self.num_technicians + 1]
        self.observation_space = gym.spaces.MultiDiscrete(np.array((lc + sc + rm + n_tech) * self.num_machines))
        t = [*range(self.num_technicians)]
        c = [*range(self.num_components)]
        actions_single_machine = list(itertools.product(t, c))
        actions_single_machine.insert(0, (-1, -1))
        list_actions_multiple_machines = []
        for i in range(self.num_machines):
            list_actions_multiple_machines.append(actions_single_machine)
        actions_multiple_machines = np.array(list(itertools.product(*list_actions_multiple_machines)))
        idx_to_delete = []
        for i in range(actions_multiple_machines.shape[0]):
            row = actions_multiple_machines[i, :, 0]
            values_with_action = row[row >= 0]
            if len(np.unique(values_with_action)) != len(values_with_action):
                idx_to_delete.append(i)
        actions_multiple_machines = np.delete(actions_multiple_machines, idx_to_delete, axis=0)
        #Reward
        rewardBreakdown = 0
        for t in self.technicians:
            maxT2R = max(t.mt2r)
            rewardBreakdown = maxT2R if maxT2R >= rewardBreakdown else rewardBreakdown
        self.reward = [1, -rewardBreakdown, 0]

        self.list_actions = actions_multiple_machines.copy()
        self.action_space = gym.spaces.Discrete(self.list_actions.shape[0])
        self.machines_names = []
        self.allMachines_steps = {}
        for m in self.machines:
            self.machines_names.append(f"m{m.id}")
            self.allMachines_steps[m.id] = []
        #self.fig, self.gnt = plt.subplots()
        #self.colors_states = ['#2dca1cff', '#a50000ff', '#00b3dacc', '#9927f599','#27eaf5ff', '#ff5bf0cc', '#c65900ff', '#aef527cc', '#6d0000ff']



    def get_observation(self):
        obs = []
        for m in self.machines:
            life_components = []
            state_componentes = []
            for lc in m.life_components:
                life_components.append(lc)
            for sc in m.state_components:
                state_componentes.append(sc)
            obs = obs + life_components + state_componentes + [m.remaining_maintenance, m.tech_assigned + 1]
        return np.array(obs)

    def action_masks(self):
        final_mask = np.array([True]*self.list_actions.shape[0])
        for i, m in enumerate(self.machines):
            tmp_mask = []
            if m.state == 2:
                for j in range(self.list_actions.shape[0]):
                    tmp_mask.append(np.array_equal(self.list_actions[j, i, :], [-1, -1]))
            else:
                for j in range(self.list_actions.shape[0]):
                    if np.array_equal(self.list_actions[j, i, :], [-1, -1]):
                        tmp_mask.append(True)
                    else:
                        tmp_mask.append(self.technicians[self.list_actions[j, i, 0]].state)
            final_mask = np.logical_and(final_mask, tmp_mask)
        return final_mask

    def reset(self):
        self.current_step = 0
        for t in self.technicians:
            t.reset()
        for m in self.machines:
            m.reset()
        return self.get_observation()

    def step(self, action):
        rew = 0
        done = False
        for i, m in enumerate(self.machines):
            op_correct = m.assign_tech(self.technicians,
                                       self.list_actions[action][i][0],
                                       self.list_actions[action][i][1])
            if op_correct:
                rew = rew + self.reward[m.step(self.technicians)]
            else:
                obs = self.get_observation()
                rew = 0
                done = True
                return obs, rew, done, {}

        obs = self.get_observation()
        self.current_step += 1
        if self.current_step >= self.max_episode_length:
            done = True
        return obs, rew, done, {}

    #def render(self, mode="human"):
        # self.initGantt()
        # for i, m in enumerate(self.machines):
        #     s = 0
        #     steps = []
        #     colors = []
        #     colorsState = []
        #     for h in m.history:
        #         steps.append((s, 1))
        #         s += 1
        #         colors.append(self.colors_states[h+2])
        #         if h >= 0:
        #             colorsState.append("#ffc700ff")
        #         else:
        #             colorsState.append(self.colors_states[h+2])
        #     self.gnt.broken_barh(steps, ((i + 1) * 10, 9), facecolors=tuple(colorsState))
        #     self.gnt.broken_barh(steps, ((i + 1) * 10 + 4, 2), facecolors=tuple(colors))
        #
        # if mode == "human":
        #     plt.show()
        #     #self.fig.canvas.draw()
        # if mode == "rgb_array":
        #     #plt.show()
        #     self.fig.canvas.draw()
        #     data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        #     data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        #     return data

    def close(self):
        #pygame.quit()
        return True

    def initGantt(self):
        self.fig, self.gnt = plt.subplots()
        # Setting Y-axis limits
        self.gnt.set_ylim(0, 20+10*self.num_machines)
        # Setting X-axis limits
        self.gnt.set_xlim(0, self.max_episode_length)
        self.gnt.set_xlabel('Timesteps')
        self.gnt.set_ylabel('Machines')
        self.gnt.set_yticks(list(range(15, self.num_machines*10+6, 10)))
        # Labelling tickes of y-axis
        self.gnt.set_yticklabels(self.machines_names)
        # Setting graph attribute
        self.gnt.grid(True)