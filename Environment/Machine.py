import numpy as np
import reliability as rlb
from typing import List
from Environment.Technician import Technician


class Machine:
    def __init__(self, idx, failure_dist):
        self.id = idx
        self.state = 0
        self.failure_dist = []
        self.num_failures = len(failure_dist)
        self.life_components = np.array([0] * self.num_failures)
        self.failure_time = np.array([0] * self.num_failures)
        self.state_components = np.array([0] * len(failure_dist))
        self.remaining_maintenance = 0
        self.component_maintenance = -1
        self.tech_assigned = -1
        self.maintenance_action = -1
        self.reward = [1, -1, 0]
        self.assignment_success = 0
        self.text_state = ["  working  ", " breakdown ", "maintenance"]
        self.colors_state = [(102, 255, 0, 75), (180, 0, 0, 75), (255, 210, 0, 75)]
        self.img_position = (0, 0)
        self.history = []
        np.random.seed(0)
        for i, (alpha, beta) in enumerate(failure_dist):
            self.failure_dist.append(rlb.Distributions.Weibull_Distribution(alpha=alpha, beta=beta))
            self.failure_time[i] = int(np.ceil(self.failure_dist[-1].random_samples(1) + np.finfo(float).eps))

    def step(self, tech: List[Technician]):
        if self.state == 0:
            self.life_components += 1
            any_fail = np.argwhere(self.failure_time - self.life_components <= 0).T.tolist()[0]
            self.state_components[any_fail] = 1  # Breakdown
            if len(any_fail) > 0:
                self.state = 1
            self.history.append(-2)
        elif self.state == 1:  # Breakdown
            self.history.append(-1)
            pass
        elif self.state == 2:  # Maintenance
            self.remaining_maintenance -= 1
            self.history.append(self.tech_assigned)
            if self.remaining_maintenance <= 0:
                #print(self.maintenance_action)
                self.failure_time[self.component_maintenance] = int(
                    np.ceil(self.failure_dist[self.component_maintenance].random_samples(1) + np.finfo(float).eps))
                self.life_components[self.component_maintenance] = 0
                self.component_maintenance = -1
                self.maintenance_action = -1
                # Reset technician parameters
                tech[self.tech_assigned].set_state_machine(True, -1)
                self.tech_assigned = -1
                # Check if there is more failures to be fix
                self.state_components *= 0
                any_fail = np.argwhere(self.failure_time - self.life_components <= 0).T.tolist()[0]
                self.state_components[any_fail] = 1  # Breakdown
                self.state = 1 if len(any_fail) > 0 else 0
        return self.state

    def assign_tech(self, tech: List[Technician], idx_tech: int, action: int):
        self.maintenance_action = action
        self.assignment_success = True
        if action >= 0:
            #print(action)
            if self.state <= 1 and tech[idx_tech].state:
                self.state = 2
                self.tech_assigned = idx_tech
                self.remaining_maintenance = tech[idx_tech].mt2r[action]
                self.component_maintenance = action
                tech[idx_tech].set_state_machine(False, self.id)
                self.assignment_success = True
            else:
                self.assignment_success = False
        return self.assignment_success

    def reset(self):
        self.state = 0
        self.life_components = np.array([0] * len(self.failure_dist))
        self.state_components = np.array([0] * len(self.failure_dist))
        self.remaining_maintenance = 0
        self.tech_assigned = -1
        self.maintenance_action = -1
        self.component_maintenance = -1
        self.failure_time = np.array([0] * self.num_failures)
        self.assignment_success = True
        self.history = []
        for i, d in enumerate(self.failure_dist):
            self.failure_time[i] = int(np.ceil(self.failure_dist[i].random_samples(1) + np.finfo(float).eps))
