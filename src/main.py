# Rllib docs: https://docs.ray.io/en/latest/rllib.html
# Malmo XML docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import math

import gym, ray
from numpy.random.mtrand import f
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
from ray.rllib.agents import dqn
import collections

from constants import ProblemType

# Problem setup parameters

# changes terrain (flat vs hill side)
problem_type = ProblemType.hill

# whether or not the agent is using DiscreteMovementCommands
discrete_moves = False

# randomly spawn Ghast in four directions around agent (north, west, south, east)
# only working for flat world terrain.
random_spawn = False

# reward for placing blocks
reward_blocks = True

# reward for facing ghast
reward_facing_ghast = True
# reward for placing blocks amount
reward_mult = 5

# Compresses the observation space by not giving the agent its yaw value
# and instead changing the arrangement of nearby blocks to match
# current yaw value.
# If set to False, this will also provide the agent's pitch value to the agent.
yaw_obs_simplifier = False


# Adds the coordinates of the Ghast to the observation space
# This will help the agent to learn the position of the Ghast
# and also with the help of the reward for facing the Ghast
obs_ghast_coordinate = True

# Gives the agent observation of its pitch value.
obs_pitch = True

mob_type = "Ghast"

# Verify that the parameters will result in an environment that has been
# configured. Not all combinations of parameters have been set up properly,
# so this provides a scalable way of avoiding those combinations.
parameter_not_configured_msg = "{} parameter not configured for current world type. Should be left as: {}."
if problem_type is ProblemType.flat:
    pass
elif problem_type is ProblemType.hill:
    assert not random_spawn,  parameter_not_configured_msg.format("random_spawn", False)

if yaw_obs_simplifier:
    assert reward_facing_ghast, "yaw_obs_simplifier being True prevents any additional observations, including [reward_facing_ghast]."

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
        self.block_quantity = 63
    
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
        
        obs_space_tmp = self.obs_array_length()
        
        self.observation_space = Box(-self.size * 2, self.size * 2, shape=(obs_space_tmp, ), dtype=np.float32)

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
        self.fireballs = collections.defaultdict(dict)
        self.enemy = collections.defaultdict(dict)

        self.last_damage_taken = 0
        self.last_block_count = 0
        self.facing_ghast_reward = 0
        # agent starts looking down
        self.looking_down = True

    
    def obs_array_length(self):
        """Returns the length of the observation array."""
        length = self.obs_height * self.obs_size * self.obs_size
        if not yaw_obs_simplifier:
            length += 1

            if reward_facing_ghast:
                length += 1
            
            if obs_ghast_coordinate:
                length +=3 
            
            if obs_pitch:
                length += 1

        return length

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

        self.facing_ghast_reward = 0
        self.last_block_count = 0
        self.looking_down = True

        self.fireballs.clear()

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

    def extract_obs_running(self, world_state) -> dict:
        """Returns the world state's observation dictionary given certain conditions.
        Returns None if the conditions do not apply."""
        if world_state.is_mission_running and \
        world_state.number_of_observations_since_last_state > 0:
            return self.extract_observations(world_state)
        else:
            return None

    def step_reward_blocks(self, world_state, reward_mult: int = 1) -> int:
        """Mutates self.last_block_count.
        Returns positive value based on how many blocks were placed in last iteration."""
        observations = self.extract_obs_running(world_state)
        if observations is None or observations['Life'] <= 0:
            return 0
        
        blocks_used = self.block_quantity - observations['InventorySlot_0_size']
        reward = 0

        if blocks_used > self.last_block_count:
            reward = (blocks_used - self.last_block_count) * reward_mult
            self.last_block_count = blocks_used

        return reward
    
    def is_facing_ghast(self, world_state) -> Optional[bool]:
        """Whether or not the agent is looking in the direction (left-right plane)
        of a Ghast.
        
        Returns None if unable to find both Steve and the Ghast, or if mission
        is not running."""
        # Get Entities observation
        observations = self.extract_obs_running(world_state)
        if observations is None:
            return None
        entitySight = observations["entitySight"]
        found_steve = False
        found_ghast = False

        for entity in entitySight:
            id = entity['id']
            name = entity['name']

            if name == "SteveTheBuilder":
                x_steve = entity["x"]
                z_steve = entity["z"]
                yaw_steve = entity["yaw"]
                found_steve = True
            
            elif name == mob_type:
                x_ghast = entity["x"]
                z_ghast = entity["z"]
                yaw_ghast = entity["yaw"]
                found_ghast = True
        
        if (found_steve and found_ghast):
            # Find the yaw the agent need to be for looking at the ghast
            hypothenus_distance = math.sqrt( (x_steve-x_ghast)**2 + (z_steve-z_ghast)**2 )
            adjacent_distance = abs(x_ghast-x_steve)
            alpha = math.degrees(math.acos(adjacent_distance/hypothenus_distance))
            if z_ghast > z_steve:
                if(x_ghast>x_steve):
                    yaw = 270 + alpha
                else:
                    yaw = 90 - alpha                

            else:
                if(x_ghast>x_steve):
                    yaw = 270 - alpha
                else:
                    yaw = 90 + alpha

            return yaw<=yaw_steve+70 and yaw>=yaw_steve-70


    def step_reward_facing_ghast(self,world_state) -> int:
        reward = 0
        is_facing = self.is_facing_ghast(world_state)

        if is_facing is not None:
            if is_facing:
                reward= reward + 2
                #print("looking at ghast")
            else:
                reward= reward - 0.5
                #print("not looking at ghast")
        
        return reward


    def step_reward_damage(self, world_state) -> int:
        """Mutates self.last_damage_taken.
        Returns negative value based on how much damage taken."""
        observations = self.extract_obs_running(world_state)
        if observations is None:
            return 0

        # DamageTaken observation resets only when launchClient.bat is restarted,
        # however, self.last_damage_taken resets every time main.py is run,
        # so only calculate reward for damage taken after the first few steps.
        new_damage_taken = observations['DamageTaken']
        if len(self.steps) <= 1 and self.episode_step < 7:
            reward = 0
        else:
            reward = - ((new_damage_taken - self.last_damage_taken) // 4)
        self.last_damage_taken = new_damage_taken

        return reward

    def step_reward(self, world_state) -> float:
        """Mutates self.episode_return (adds on rewards since last world state was taken).
        Returns rewards since last world state was taken."""
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        blocks_placed = False
        facing_ghast = False
        reward += self.step_reward_damage(world_state)
        if reward_blocks:
            if self.step_reward_blocks(world_state):
                blocks_placed = True
            reward += self.step_reward_blocks(world_state)
        if reward_facing_ghast:
            if self.step_reward_facing_ghast(world_state):
                facing_ghast = True
            reward += self.step_reward_facing_ghast(world_state)
        if blocks_placed and facing_ghast:
            reward += 2
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
        mob_type_xml = mob_type  
        return f"<DrawEntity x='{x}' y='{y}' z='{z}' type='{mob_type_xml}'/>"

    def get_mission_xml(self):
        if problem_type is ProblemType.flat:
            if random_spawn:
                x = self.enemy_spawn_distance if randint(2) else -self.enemy_spawn_distance
                z = self.enemy_spawn_distance if randint(2) else -self.enemy_spawn_distance
            else:
                x = self.enemy_spawn_distance
                z = self.enemy_spawn_distance
        
        if problem_type is ProblemType.hill:
            x = self.enemy_spawn_distance
            z = self.enemy_spawn_distance
        enemy_starting_location = (x, 1, z)

        if problem_type is ProblemType.flat:
            draw_terrain = "<DrawCuboid x1='{}' x2='{}' y1='2' y2='30' z1='{}' z2='{}' type='air'/>".format(-self.size, self.size, -self.size, self.size) +  "<DrawCuboid x1='{}' x2='{}' y1='1' y2='1' z1='{}' z2='{}' type='stone'/>".format(-self.size, self.size, -self.size, self.size)
            agent_spawn = f'<Placement x="0" y="2" z="0" pitch="45" yaw="0"/>'
        elif problem_type is ProblemType.hill:
            # clear out extra space to try to remove any Ghasts that go really
            # far
            air_size = self.size * 2
            draw_terrain = "<DrawCuboid x1='{}' x2='{}' y1='2' y2='100' z1='{}' z2='{}' type='air'/>".format(-air_size, air_size, -air_size, air_size)
            for i in range(1, 21):
                layer_size = 21 - i
                draw_terrain += f"<DrawCuboid x1='{-layer_size}' x2='{layer_size}' y1='1' y2='{i}' z1='{-layer_size}' z2='{layer_size}' type='{self.player_block}'/>"
            agent_spawn_y = randint(2, 22)
            agent_spawn_z = 22 - agent_spawn_y
            agent_spawn = f'<Placement x="0" y="{agent_spawn_y}" z="{agent_spawn_z}" pitch="45" yaw="0"/>'
            enemy_starting_location = (x+y for x, y in zip(enemy_starting_location, (1, agent_spawn_y, agent_spawn_z)))

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
                                draw_terrain + \
                                f"{self.get_enemy_xml(*enemy_starting_location)}" + \
                                '''<DrawBlock x='0'  y='2' z='0' type='air' />
                                <DrawBlock x='0'  y='1' z='0' type='stone' />
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>

                    <AgentSection mode="Survival">
                        <Name>SteveTheBuilder</Name>
                        <AgentStart>''' + \
                            agent_spawn + \
                            '''<Inventory>''' + \
                                f'<InventoryItem slot="0" type="{self.player_block}" quantity="{self.block_quantity}"/>' + \
                            '''</Inventory>
                        </AgentStart>
                        <AgentHandlers>''' + \
                            movement + \
                            '''<ObservationFromFullStats/>
                            <ObservationFromFullInventory/>
                            <ObservationFromRay/>
                            <ObservationFromGrid>
                                <Grid name="nearbyVolume">''' + \
                                    f'<min x="-{str(int(self.obs_size/2))}" y="{obs_low_y}" z="-{str(int(self.obs_size/2))}"/>' + \
                                    f'<max x="{str(int(self.obs_size/2))}" y="{obs_high_y}" z="{str(int(self.obs_size/2))}"/>' + \
                                '''</Grid>
                            </ObservationFromGrid>
                            <ObservationFromNearbyEntities>
                                <Range name="entitySight" xrange="15" yrange="25" zrange="50" />
                            </ObservationFromNearbyEntities>
                            
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
        # my_mission.forceWorldReset();
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
        obs_tmp = self.obs_array_length()
        obs = np.zeros((obs_tmp, ))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                observations = self.extract_observations(world_state)

                # Get grid observation
                grid = observations.get('nearbyVolume')

                # avoid KeyError issue by checking if observations has values.
                if grid is None:
                    print(f"Encountered a KeyError issue on step {self.steps[-1]}.")
                    break
                for i, x in enumerate(grid):
                    obs[i] = x == self.player_block

                yaw = observations['Yaw']
                # from https://edstem.org/us/courses/14172/discussion/863158 suggestion in comments
                if yaw < 0:
                    yaw += 360

                # decrement this by 1 every time used, so no overwriting other information
                extra_val_index = -1

                if yaw_obs_simplifier:
                    # Rotate observation with orientation of agent
                    obs = obs.reshape((self.obs_height, self.obs_size, self.obs_size))

                    if yaw >= 225 and yaw < 315:
                        obs = np.rot90(obs, k=1, axes=(1, 2))
                    elif yaw >= 315 or yaw < 45:
                        obs = np.rot90(obs, k=2, axes=(1, 2))
                    elif yaw >= 45 and yaw < 135:
                        obs = np.rot90(obs, k=3, axes=(1, 2))

                    obs = obs.flatten()
                else:
                    # make yaw a decimal value so fits inside observation space box.
                    obs[extra_val_index] = yaw/360
                    extra_val_index -= 1

                    if reward_facing_ghast:
                        ghast_index = extra_val_index
                        extra_val_index -= 1
                        
                        facing_ghast = self.is_facing_ghast(world_state)
                        if facing_ghast is not None:
                            obs[ghast_index] = 1 if facing_ghast else 0
                        
                    # TODO coordinate code probably needs to be relative to the agent.
                    # Imagine if the Ghast is in a certain location. The agent
                    # could be in many different locations relatively, and this changes
                    # the situation pretty drastically. However, there is no change
                    # in observation.
                    if obs_ghast_coordinate:
                        ghast_coordinate = observations["entitySight"]

                        for entity in ghast_coordinate:
                            if entity['name'] == mob_type:
                                obs[extra_val_index] = entity["x"]/2
                                extra_val_index -= 1

                                obs[extra_val_index] = entity["y"]/2
                                extra_val_index -= 1

                                obs[extra_val_index] = entity["z"]/2
                                extra_val_index -= 1
                                #print("coordinates:",entity["x"],entity["y"],entity["z"])
                                break

                    if obs_pitch:
                        pitch = observations["Pitch"]

                        obs[extra_val_index] = pitch/90
                        extra_val_index -= 1
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
