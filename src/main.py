# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# Malmo XML docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

import gym, ray
from numpy.random.mtrand import f
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
from ray.rllib.agents import dqn

# Problem setup parameters

# whether or not the agent is using DiscreteMovementCommands
discrete_moves = True

# randomly spawn Ghast in four directions around agent (north, west, south, east)
random_spawn = True



class SteveTheBuilder(gym.Env):

    def __init__(self, env_config):
        self.discrete_moves = discrete_moves

        # Static Parameters
        self.size = 50
        self.enemy_spawn_distance = 4
        self.obs_size = 3
        self.obs_height = 3
        self.max_episode_steps = 100 if self.discrete_moves else 300
        self.log_frequency = 10

        if self.discrete_moves:
            self.action_dict = {
                0: 'turn 1',  # Turn 90 degrees to the right.
                1: 'turn -1',  # Turn 90 degrees to the left.
                2: 'look 1', # Pitch 45 degrees down.
                3: 'look -1', # Pitch 45 degrees up.
                4: 'use' # Place a block.
            }

        # Rllib Parameters
        if self.discrete_moves:
            self.action_space = Discrete(len(self.action_dict))
        else:
            self.action_space = Box(low=np.array([-1.0, -1.0, -1.0]),
                            high=np.array([1.0, 1.0, 1.0]))
        
        self.observation_space = Box(0, 1, shape=(self.obs_height * self.obs_size * self.obs_size, ), dtype=np.float32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # Constants
        self.player_block = "cobblestone"

        # Dynamic Parameters
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

        self.last_damage_taken = 0
        # agent starts looking down
        self.looking_down = True

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        self.looking_down = True

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs

    def extract_observations(self, world_state):
        """Returns latest observation dictionary from world state.
        Returns None if mission is not running or no new observations."""
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        return observations

    def step_reward_damage(self, world_state) -> int:
        """Mutates self.last_damage_taken. Returns negative value based on how much damage taken."""
        if world_state.is_mission_running and \
        world_state.number_of_observations_since_last_state > 0:
            observations = self.extract_observations(world_state)
        else:
            return 0
        new_damage_taken = observations['DamageTaken']
        reward = - ((new_damage_taken - self.last_damage_taken) // 4)
        self.last_damage_taken = new_damage_taken

        return reward

    def step_reward(self, world_state) -> float:
        """Mutates self.episode_return (adds on rewards since last world state was taken).
        Returns rewards since last world state was taken."""
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()

        reward += self.step_reward_damage(world_state)

        self.episode_return += reward

        return reward

    def step_action(self, action):
        command = self.action_dict[action]

        # limiting agent to only looking down or up (ground block layer and 1 above that)
        down_cmd = command == "look 1"
        up_cmd = command == "look -1"
        looking_too_far_down = down_cmd and self.looking_down
        looking_too_far_up = up_cmd and not self.looking_down

        if not (looking_too_far_down or looking_too_far_up):
            if down_cmd or up_cmd:
                # if the agent inputs a down command, it must be the case the agent
                # is not looking down. And vice-versa.
                self.looking_down = not self.looking_down
            self.agent_host.sendCommand(command)
            time.sleep(.2)
            self.episode_step += 1

    def step_continuous_action(self, action: List[float]) -> None:
        turn_val, pitch_val, use_val = action
        use_val = 1 if use_val > 0 else 0

        self.agent_host.sendCommand(f'turn {turn_val}')
        self.agent_host.sendCommand(f'pitch {pitch_val}')
        self.agent_host.sendCommand(f'use {use_val}')

        time.sleep(.2)
        self.episode_step += 1

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        if self.discrete_moves:
            self.step_action(action)
        else:
            self.step_continuous_action(action)

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state)

        # Get Done
        done = not world_state.is_mission_running

        # Get Reward
        reward = self.step_reward(world_state)
        
        return self.obs, reward, done, dict()

    def get_enemy_xml(self, x, y, z) -> str:
        # for mob types, see:
        # https://microsoft.github.io/malmo/0.30.0/Schemas/Types.html#type_EntityTypes
        mob_type = "Ghast"
        return f"<DrawEntity x='{x}' y='{y}' z='{z}' type='{mob_type}'/>"

    def get_mission_xml(self):
        block_quantity = 63

        if random_spawn:
            x = self.enemy_spawn_distance if randint(2) else -self.enemy_spawn_distance
            z = self.enemy_spawn_distance if randint(2) else -self.enemy_spawn_distance
        else:
            x = self.enemy_spawn_distance
            z = self.enemy_spawn_distance
        enemy_starting_location = (x, 1, z)

        if self.discrete_moves:
            movement = "<DiscreteMovementCommands/>"
        else:
            movement = "<ContinuousMovementCommands/>"

        time_reward = "<RewardForTimeTaken initialReward='1' delta='1' density='PER_TICK' />"

        obs_low_y = -1
        obs_high_y = 1

        assert self.obs_height == obs_high_y - obs_low_y + 1, f"SteveTheBuilder.get_mission_xml: [self.obs_height] value of {self.obs_height} is not equal to {obs_high_y - obs_low_y + 1}, which is [obs_high_y] - [obs_low_y] + 1."

        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                        <Summary>SteveTheBuilder</Summary>
                    </About>

                    <ServerSection>
                        <ServerInitialConditions>
                            <Time>
                                <StartTime>12000</StartTime>
                                <AllowPassageOfTime>false</AllowPassageOfTime>
                            </Time>
                            <Weather>clear</Weather>
                        </ServerInitialConditions>
                        <ServerHandlers>
                            <FlatWorldGenerator generatorString="3;7,2;1;"/>
                            <DrawingDecorator>''' + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='5' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size, -self.size, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='stone'/>".format(-self.size, self.size, -self.size, self.size) + \
                                f"{self.get_enemy_xml(*enemy_starting_location)}" + \
                                '''<DrawBlock x='0'  y='2' z='0' type='air' />
                                <DrawBlock x='0'  y='1' z='0' type='stone' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>SteveTheBuilder</Name>
                        <AgentStart>
                            <Placement x="0.5" y="2" z="0.5" pitch="45" yaw="0"/>
                            <Inventory>''' + \
                                f'<InventoryItem slot="0" type="{self.player_block}" quantity="{block_quantity}"/>' + \
                            '''</Inventory>
                        </AgentStart>
                        <AgentHandlers>''' + \
                            movement + \
                            '''<ObservationFromFullStats/>
                            <ObservationFromRay/>
                            <ObservationFromGrid>
                                <Grid name="nearbyVolume">''' + \
                                    f'<min x="-{str(int(self.obs_size/2))}" y="{obs_low_y}" z="-{str(int(self.obs_size/2))}"/>' + \
                                    f'<max x="{str(int(self.obs_size/2))}" y="{obs_high_y}" z="{str(int(self.obs_size/2))}"/>' + \
                                '''</Grid>
                            </ObservationFromGrid>
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                            <AgentQuitFromTouchingBlockType>
                                <Block type="bedrock" />
                            </AgentQuitFromTouchingBlockType>''' + time_reward + '''
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(1200, 720)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'SteveTheBuilder' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 3 x 5 x 5
        grid around the agent. Dimensions are [self.obs_height]
        and [self.obs_size], 3 x 5 x 5 may be deprecated.

        The agent is in the center square facing up.
        Search "<Grid name="nearbyVolume">" in mission XML to find specifics.
            Note that y is relative to the agent.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
        """
        obs = np.zeros((self.obs_height * self.obs_size * self.obs_size, ))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                observations = self.extract_observations(world_state)

                # Get observation
                grid = observations['nearbyVolume']
                for i, x in enumerate(grid):
                    obs[i] = x == self.player_block

                # Rotate observation with orientation of agent
                obs = obs.reshape((self.obs_height, self.obs_size, self.obs_size))
                yaw = observations['Yaw']

                # from https://edstem.org/us/courses/14172/discussion/863158 suggestion in comments
                if yaw < 0:
                    yaw += 360

                if yaw >= 225 and yaw < 315:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw >= 315 or yaw < 45:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw >= 45 and yaw < 135:
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                obs = obs.flatten()
                
                break
        
        return obs

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('SteveTheBuilder')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value)) 


if __name__ == '__main__':
    ray.init()
    
    if discrete_moves:
        trainer = dqn.DQNTrainer(env=SteveTheBuilder, config={
            'env_config': {},           # No environment parameters to configure
            'framework': 'torch',       # Use pyotrch instead of tensorflow
            'num_gpus': 0,              # We aren't using GPUs
            'num_workers': 0            # We aren't using parallelism
        })
    else:
        trainer = ppo.PPOTrainer(env=SteveTheBuilder, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
        })

    while True:
        print(trainer.train())
