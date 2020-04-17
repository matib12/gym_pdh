#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PDH
'''
import gym
from gym.utils import seeding
import numpy as np
from PIL import Image
from gym import spaces, logger

# Third party
import pdh  # Mateusz's library

class PdhEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.mpos_step = 1000000.
        
        self.screen_width = 600
        self.screen_height = 400
        self.data = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)  # For render
        self.iterator = 0
        
        # Real values from MCIR fit
        self.reflectivity = 0.9988144404882143
        self.FSR = 550.0e6   # in Hz
        self.Pc = 1.         # indicates maximum of the trans and refl signals
        self.Ps = 0.1        # fraction of Pc power
        self.Omega = 80.0e6  # in Hz
        self.ampl = -0.0402761612065936
        
        self.sideband_hight = (self.Pc * self.Ps)*1.01  # To exclude SB

        # Set cavity resonance in a random position equvalent to set
        # mpos to a random position based on self.Omega
        self.over_counter = 0
        self.gym_pdh = pdh.pdh_cavity(self.reflectivity, self.FSR, self.Pc, self.Ps, self.Omega)

        # Stores:
        # 1 error
        # 2 err derivative
        # 3 refl
        # 4 refl deriv.
        # 5 trans
        # 6 trans deriv.
        # 7 mpos
        self.state = None
        
        # Angle at which to fail the episode
        self.mpos_threshold = 1.2 * self.FSR  # Resonances must stay in this range
        
        # Limit set to 2 *  so that failing observation is still within bounds
        high = np.array([2.,                         # 1 error
                         np.finfo(np.float32).max,   # 2 err derivative
                         self.Pc * 2,                # 3 refl
                         np.finfo(np.float32).max,   # 4 refl deriv.
                         self.Pc * 2,                # 5 trans
                         np.finfo(np.float32).max,   # 6 trans deriv.
                         self.FSR * 4.],             # 7 mpos
                        dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.action_space = spaces.Discrete(2)  # From cartPole example. Discrete move to the right/left
        #self.action_space = spaces.Box(  # From MountainCar env. Continous control
        #    low=self.min_action,
        #    high=self.max_action,
        #    shape=(1,),
        #    dtype=np.float32
        #)
        
        self.reward_range = (0., self.Pc)  # From core.py github code
        
        self.seed()  # initialize random number generator self.np_random
        
        self.steps_beyond_done = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Observation: 
            Type: Box(4)
                Num Observation                   Min         Max
                0   Error signal Position         -1.0        1.0
                1   Error signal Velocity         -Inf        Inf
                2   Reflection signal Position    0.0         1.0 (Pc)
                3   Reflection signal Velocity    -Inf        Inf
                4   Transmission signal Position  0.0         1.0 (Pc)
                5   Transmission signal Velocity  -Inf        Inf

        
        Actions:
            Type: Continous(2)
            Num    Action
            0      Subtract delta z from output signal
            1      Add delta z to output signal

        Episode Termination:
            Episode length is greater than 200
            Solved Requirements (Locked above 90% for 20? iterations)
            Reaches >90% but then falls?
            Output more than +/-FSR. No reward

        Parameters
        ----------
        action :
            set cavity mirror position.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        err, err_dot, refl, refl_dot, trans, trans_dot, mpos = state
        
        step = self.mpos_step if action==1 else -self.mpos_step        
        
        mpos = mpos + step
        mposrad = 2.*np.pi*mpos
        n_err = self.gym_pdh.cavity_err(mposrad)
        err_dot = n_err - err
        n_refl = self.gym_pdh.cavity_refl(mposrad)
        refl_dot = n_refl - refl
        n_trans = self.gym_pdh.cavity_trans(mposrad)
        trans_dot = n_trans - trans
        
        self.state = (n_err,err_dot,n_refl,refl_dot,n_trans,trans_dot,mpos)
        done = mpos < -self.mpos_threshold \
               or mpos > self.mpos_threshold
        done = bool(done)
        
        """
        Ideas:
        1. Reward is proportional to the transmission level from 0. to 1.
        2. Reward is zero if Refl is less than Pc*Ps. Locked on a sideband?
        3. Reward takes into account the derivative of refl.
        """
        r = self.state[4]
        #if r > self.sideband_hight:
        #    return r # + self.state[3]  # If uncomment chenck reward range in __init__
        #else:
        #    return 0.0
        
        if not done:
            reward = r
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        
        return np.array(self.state), reward, done, {}

    def reset(self):
        self._show()

        mpos = self.np_random.uniform(low=-self.FSR, high=self.FSR)
        noises = self.np_random.uniform(low=-0.05, high=0.05, size=(7,))  # additional noise to mpos which is already random generated
        noises = noises + [self.gym_pdh.cavity_err(mpos), 0.,\
                           self.gym_pdh.cavity_refl(mpos), 0.,\
                           self.gym_pdh.cavity_trans(mpos), 0.,\
                           mpos]
        self.state = noises
        
        self.data = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)  # For render
        self.iterator = 0
        
        self.steps_beyond_done = None

        return np.array(self.state)
        
    def _show(self):
        try:
            img = Image.fromarray(self.data, 'RGB')  # of into close()
            #img.save('my.png')
            img.show()
        except:
            print("Cannot visualize")

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            index = self.iterator
            
            if index >= self.screen_width:
                logger.warn("You are calling 'render()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                
            hight = self.screen_height-1
            
            column_wd = 1
        
            world_width = self.screen_width/column_wd  # number of columns(steps) in the visualisation
            
            err_pos = np.interp(self.state[0], (-1., 1.), (0, hight))
            refl_pos = np.interp(self.state[2], (0., 1.), (0, hight))
            trans_pos = np.interp(self.state[4], (0., 1.), (0, hight))
            mpos_pos = np.interp(self.state[6], (-self.mpos_threshold, self.mpos_threshold), (0, hight))
            
            self.data[int(err_pos), index] = [255, 0, 0]  # error Red pixel in index column and err_pos row
            self.data[int(refl_pos), index] = [0, 255, 0]  # reflection Blue
            self.data[int(trans_pos), index] = [0, 0, 255]  # reward but not transmission Green
            self.data[int(mpos_pos), index] = [255, 255, 255]  # mpos White
            
            self.iterator += 1
            #if self.viewer is None:
            #    from gym.envs.classic_control import rendering
            #    self.viewer = rendering.Viewer(screen_width, screen_height)
        elif mode == 'human':
            pass
        
    def close(self):
        self._show()

    #def info(self):
    #    pass
