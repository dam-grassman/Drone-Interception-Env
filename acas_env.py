from re import A
import sys

sys.path.insert(0, "..")

import gym
from gym import spaces

import numpy as np

from numpy.random import rand
from math import pi
import cv2
from utils import *

ANGLE_STEP = pi / 6
COST_STRING = {
    0: "COC",
    1: "WR",
    2: "WL",
    3: "R",
    4: "L",
}

class AntiCollisionSystem:
    
    """
    Anti Collision system class to deal with both the ACAS table 
    or a surrogate model (a scikit-learn's RF in this case)
    
    """
    
    def __init__(self, cost_table=None, surrogate=None, scaler=None):
        """
        
        :param:
            cost_table: table ACAS from deel datasets if access allowed
            surrogate:  a surrogate model (any scikit-learn model should work)
            scaler:     a scaler from scikit-learn to transform the input before prediction. Optional.
        """

        self.cost_table=cost_table 
        self.surrogate=surrogate
        self.scaler=scaler
        
    def get(self, features, acas_state):
        """
        :param:
            features:   features dictionnary whose keys=[range, theta, psi, ownspeed, intrspeed]
            acas_state: current acas state
        
        :return:
            cost_state: advisories from either the ACAS table/surrogate.
        """
        if self.cost_table is not None :
            c = self.cost_table.get_cost(COST_STRING[acas_state], features)
            cost_state = np.argmin(c)
        elif self.surrogate is not None:
            x = np.array([features["range"], features["theta"], features["psi"], features["ownspeed"],  features["intrspeed"], acas_state])
            x = x.reshape(1,-1)
            if self.scaler is not None :
                x = self.scaler.transform(x)
            cost_state = self.surrogate.predict(x)[0]
        return cost_state  


#import deel.datasets
# Load ACAS Xu Table
#print("LOAD COST TABLE of ACAS-Xu")
#COST_TABLE = deel.datasets.load("acas-xu")
#ACS_acas = AntiCollisionSystem(cost_table=COST_TABLE)




class AcasEnv(gym.Env):
    """
    ACAS environment for gym.

    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 acs,
                 gameplay_mode = False,
                 vtarget=400, 
                 vattacker=300, 
                 tolerance=1000, 
                 randomness=0,
                 win_zone=[None, None], 
                 loose_zone=[None, None], 
                 attacker_init=[None, None], 
                 target_init=[None, None]):
        
        """
        Initialization of the environment.

        :param:
            acs:            Anti-collision system 
            gameplay_mode:  If True, is playable using gym's play fn
            vtarget:        Speed of the target drone
            vattacker:      Speed of the attacker drone
            tolerance:      Tolerance distance to win/loose zone area
            randomness:     Randomness factor to use when reseting the env
            loose_zone:     Position of the loose zone
            win_zone:       Position of the win zone
            attacker_init:  Initial position of the attacker drone
            target_init:    Initial position of the target drone
        """
        assert acs in ["acas", "surrogate"]
        if acs=="acas":
            import deel.datasets
            #Load ACAS Xu Table
            print("LOAD COST TABLE of ACAS-Xu")
            COST_TABLE = deel.datasets.load("acas-xu")
            self.acs = AntiCollisionSystem(cost_table=COST_TABLE)
        else :
            import joblib
            loaded_rf = joblib.load("./MODELS/surrogate_model/acas_uncompressed.joblib")
            loaded_rf.verbose = 0
            loaded_rf.n_jobs=1
            from pickle import load
            scaler = load(open('./MODELS/surrogate_model/scaler.pkl', 'rb'))
            self.acs = AntiCollisionSystem(surrogate=loaded_rf, scaler=scaler)
        self.gameplay_mode = gameplay_mode
        self.randomness = randomness
        self.tolerance = tolerance
        
        self.vtarget = vtarget
        self.vattacker = vattacker
        self.state = [None] * 13
        
        self.win_zone = win_zone
        self.loose_zone = loose_zone
        self.attacker_init = attacker_init
        self.target_init = target_init
        
        self.action_space = spaces.Box(
            np.array([-500,-pi]), 
            np.array([500,pi])
        )
        self.observation_space = spaces.Box(
            np.array([0, 0, 0, -pi ,0, 0, 0, -pi, 0, 0, 0, 0, 0]),
            np.array([10000, 10000, 1000, pi, 10000, 10000, 1000,pi, 5, 10000, 10000, 10000, 10000])
        )
    
        self.episode = 0
        self.state = self.reset()

    def get_keys_to_action(self):
        """
        
        :return:
            keys_to_action: dictionnary mapping keyboads event to actions
        """
        keys_to_action= {
            (ord('w'),): ('SPEED_UP'),
            (ord('s'),): ('SLOW_DOWN'),
            (ord('a'),): ('LEFT'),
            (ord('d'),): ('RIGHT'),
            (ord('a'), ord('w')): ("SPEED_UP", "LEFT"),
            (ord('a'), ord('s')): ("SLOW_DOWN", "LEFT"),
            (ord('d'), ord('w')): ("SPEED_UP", "RIGHT"),
            (ord('d'), ord('s')): ("SLOW_DOWN", "RIGHT")}
                 
        return keys_to_action

    def increase_random(self):
        """
        Increase Randomness of the initial draw of posistions.
        
        """
        self.randomness += 0.05
        self.randomness = min(self.randomness, 1)
        print("Env is Increasing Random! New value is %.2f" % self.randomness)
    
    def check(self):
        """
        Checking if the game is over.

        :return:
            out: 1 if attacker wins, 2 if target wins, 0 else
        """
        al, self.rl = get_polar(self.state[4], self.state[5], self.loose_zone[0], self.loose_zone[1])
        aw, self.rw = get_polar(self.state[4], self.state[5], self.win_zone[0], self.win_zone[1])
        self.acas_state = self.state[8]
        
        if self.rw < self.tolerance:
            return 1
        elif self.rl < self.tolerance:
            return 2
        else:
            return 0

    def compute_target_heading(self):
        """
        Computation of target heading

        :return: new angle for target (in rad please)
        """
        # Polar angle from Target(x,y) to Attacker (x,y)
        a, ro = get_polar(self.state[4], self.state[5], self.state[0], self.state[1])
        # Theta = angle between target and attacker relative to the heading of the target
        theta = wrap_to_pi(a - self.state[7])
        # Psi = Angle of attacker angle relative to angle of target 
        psi = wrap_to_pi(self.state[3] - self.state[7])

        vi = {
            "vertical_tau": 0,  # We only consider 0. Target & Attacker in the same altitude
            "range": ro,  # Distance between target and attacker
            "theta": -theta,  # Angle between target and attacker relative to the heading of the target
            "psi": -psi,  # Relative heading between target and attacker
            "ownspeed": self.state[6],  # Speed of the target
            "intrspeed": self.state[2],  # Speed of the attacker
        }
                
        cost_state = self.acs.get(vi, self.state[8])
        #c = self.cost_table.get_cost(COST_STRING[self.state[8]], vi)
        #cost_state = np.argmin(c)

        self.state[8] = cost_state  # Update State

        # If no conflict = go to lose zone
        if self.state[8] == 0:  # Clear of conflict. Heading toward the lossing zone
            a, _ = get_polar(
                self.state[4], self.state[5], self.loose_zone[0], self.loose_zone[1]
            )
            # Limit the rotation to ANGLE_STEP
            if wrap_to_pi(a - self.state[7]) < -ANGLE_STEP:
                a = self.state[7] - ANGLE_STEP
            elif wrap_to_pi(a - self.state[7]) > ANGLE_STEP:
                a = self.state[7] + ANGLE_STEP

            # a = 0
        # Else avoid either right, left, weak right, weak left
        elif self.state[8] == 1:  # Weak R
            a = self.state[7] - ANGLE_STEP
        elif self.state[8] == 2:  # Weak L
            a = self.state[7] + ANGLE_STEP
        elif self.state[8] == 3:  # R
            a = self.state[7] - 2 * ANGLE_STEP
        elif self.state[8] == 4:  # L
            a = self.state[7] + 2 * ANGLE_STEP

        # return angle to apply !
        return a

    def define_heading_to_loose_zone(self):
        """
        Definition of heading when not threaten. Target wants to reach the loose zone

        :return:
            target_heading: angle (rad)
        """
        a, r = get_polar(
            self.state[4], self.state[5], self.loose_zone[0], self.loose_zone[1]
        )
        return a

    def step(self, action):
        """

        :param action: array with [delta_speed, delta_angle]
        :return:
            observation: self.state
            reward: self.reward
            done: self.done
            info: self.info
        """
        
        # Check if done 
        status = self.check()
        
        # If Gameplay mode activated
        if self.gameplay_mode == True :
            speed, angle = 0,0
            if action != 0:
                speed, angle = 0,0
                if action == 0 :
                    pass
                if "SPEED_UP" in action :
                    speed += 10
                if "SLOW_DOWN" in action :
                    speed -= 10
                if "LEFT" in action :
                    angle += pi/32
                if "RIGHT" in action :
                    angle -= pi/32
            action = speed, angle

            
        # If not done
        if status == 0:

            # 1) Update Attacker's Speed
            # compute new speed & pos pf attacker considering action
            self.state[2] = self.state[2] + action[0]
            if self.state[2] > 1000:
                self.state[2] = 1000
            if self.state[2] <= 0:
                self.state[2] = 0
                
            # 2) Update Attacker's Angle
            self.state[3] = wrap_to_pi(self.state[3] + action[1])
            
            # 3) Update Attacker's Position
            # deduce new position
            delta_attacker = get_cart(self.state[2], self.state[3])
            self.state[0] += delta_attacker[0]
            self.state[1] += delta_attacker[1]

            # Add info
            self.info["traj_attacker"].append(np.array([self.state[0], self.state[1]]))
            self.counter += 1
            self.reward = 0

            # 4) Update Target's Angle 
            # compute new speed vector for target based on previous position of attacker
            self.state[7] = wrap_to_pi(self.compute_target_heading())

            # 5) Update Target's Position
            # deduce new position
            delta_target = get_cart(self.state[6], self.state[7])
            self.state[4] += delta_target[0]
            self.state[5] += delta_target[1]
            self.info["traj_target"].append(np.array([self.state[4], self.state[5]]))

            # 6) Check distance to win or loose zone
            
            # If Target is getting closer to win zone (compared to best distance): positive reward
            if self.min_rw > self.rw:
                self.reward += (self.min_rw - self.rw) / self.vtarget
            
            # if Target getting closer to loose zone: no reward 
            if self.min_rl > self.rl:
                self.reward = 0
            
            # Update best distance to win zone
            if self.min_rw > self.rw:
                self.min_rw = self.rw
            
            #Update best distance to loose zone
            if self.min_rl > self.rl:
                self.min_rl = self.rl

            # Max Step 
            if self.counter > 5000:
                self.done = True

        else:
            self.done = True
            if status == 1:
                print("attacker wins!")
                self.reward = 1
            else:
                self.reward = -1
        return self.state, self.reward, self.done, self.info

    def reset(self):
        """
        Reset all positions (drawn according to randomness factor)
        
        :return:
        """
        # Draw initial position of the attacker drone
        self.state[0], self.state[1] = get_random_pos(
            [10000 + self.randomness * 40000, 90000 - self.randomness * 40000], 
            100000 * self.randomness
        ) 
        
        self.state[2] = 0  # Initial attacker speed set to 0
        self.state[3] = 2 * pi * rand() - pi # Initial alpha attacker in range [-PI  PI]
        
        # Draw Inirial position of the target drone
        self.state[4], self.state[5] = get_random_pos(
            [10000 + self.randomness * 40000, 10000 + self.randomness * 40000], 
            100000 * self.randomness
        )
        
        self.state[6] = self.vtarget  # Initial target speed set to vtarget
        self.state[7] = 2 * pi * rand() - pi  # alpha target range [-PI  PI]
        
        self.state[8] = 0  # State of the ACAS-Xu (0 : COC, 1 : WR, 2 : WL, 3 : R, 4 : L)

        self.win_zone = get_random_pos(
                [10000 + self.randomness * 40000, 90000 - self.randomness * 40000],
                100000 * self.randomness,
        )
        
        self.loose_zone = np.array([50000,50000])

            
        self.state[9] = self.win_zone[0]
        self.state[10] = self.win_zone[1]

        self.state[11] = self.loose_zone[0]
        self.state[12] = self.loose_zone[1]

        # RW = distance from Target  to Win Zone
        _, self.rw = get_polar(
            self.state[4], self.state[5], self.win_zone[0], self.win_zone[1]
        )
        # RL = distance from Target  to Loose Zone
        _, self.rl = get_polar(
            self.state[4], self.state[5], self.loose_zone[0], self.loose_zone[1]
        )
        
        self.acas_state = 0
        self.min_rw = self.rw
        self.min_rl = self.rl

        self.counter = 0
        self.episode += 1
        self.done = False
        self.info = {
            "traj_attacker": [np.array([self.state[0], self.state[1]])],
            "traj_target": [np.array([self.state[4], self.state[5]])],
            "loose_zone": self.loose_zone,
            "win_zone": self.win_zone,
        }
        self.reward = 0

        return self.state
    
    def render(self, mode="human"):
        """
        :param mode: valid mode, must be either \"human\" or \"rgb_array\"
        
        If \"human\", open new window using pygame
        If \"rgb_array\" return the rgb array of the state image
        
        """
        assert mode in ["human", "rgb_array"]

        if mode == "human":
            cv2.imshow("Game", render_from_state(state_dict(self.state)))
            cv2.waitKey(10)
        elif mode == "rgb_array":
            return render_from_state(state_dict(self.state))