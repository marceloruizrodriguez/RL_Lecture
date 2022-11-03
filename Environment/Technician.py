from typing import List
class Technician:
    def __init__(self, mt2r: List[int], idx: int):
        # mt2r = (int) mean time to repair in hours
        # id = (int) id of the technician
        # state = (bool) True for free, False for busy
        # machine_id = (int) Technician working on machineID
        self.mt2r = mt2r
        self.id = idx
        self.state = True
        self.machine_assigned = -1

    def set_state_machine(self, new_state: bool, new_machine: int):
        self.state = new_state
        self.machine_assigned = new_machine

    def reset(self):
        self.state = True
        self.machine_assigned = -1

    def __repr__(self):
        return f"ID:{self.id} ST:{int(self.state)} MA:{self.machine_assigned}"
