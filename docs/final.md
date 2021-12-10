---
layout: default
title: Final Report
---

## Project Video
{Youtube video link goes here}
## Project Summary

Our goal is to have our agent, Steve, protect himself for as long as possible against a Ghast/multiple Ghasts by building a shelter-like block structure. A Ghast is a floating ghost-like hostile mob that shoots explosive fireballs at the player every three seconds. The Ghast is an ideal enemy for this project because it can shoot from multiple directions and angles. This sort of behavior from the Ghast, juxtaposed with the use of reinforcement learning, will allow Steve to learn how to build a more robust structure. Because minecraft is not flat, and can have hilly terrain, we also tested Steve's building potential in varying environments.

Our project uses a type of reinforcement learning called the Deep Q learning algorithm. The algorithm attempts to find the optimal action for each possible state in the situation provided. Using ObservationFromGrid and ObservationFromNearbyEntities, we can set up positive/negative rewards including whether the agent is facing the Ghast, how long Steve has survived, how much damage Steve has taken, and/or whether a block has been placed in order to determine the optimal action. The greatest challenge of this project is selecting what to reward, and how to weigh them to optimize our goal.

## Approaches

rough draft material for Approaches section:

Since the agent seemed to be metagaming when the Ghast spawned only in one direction by constructing an L-shaped wall in the direction of the Ghast at the start of the challenge, we are spawning the Ghast randomly around in the agent in four possible starting spots (north, south, east, and west of the agent). The agent is struggling more to learn in this environment. [Figure1 First prototype plus 4 random spawn] Part of the challenge is possibly inconsistent results based on observation. Say the agent builds a wall north of itself. If the Ghast is in that direction, it is helpful. However, if the Ghast is on any other direction, the Ghast will still be able to hit the agent, and in particular, if the Ghast is south of the agent the wall will actually hurt the agent by causing the agent to be stuck in fire unless it dies.

Ideally the agent could build a wall around itself completely at the start of the round, however, this may be difficult to provoke. One idea is giving the agent a reward for placing blocks. We expect this to only serve to benefit the agent, and may trivialize the problem to some degree, but it could reveal that another significant problem for the agent is figuring out where to place blocks. Adding a reward of 1 for placing blocks seems to result in some learning, although there is still significant variance in the results. [Figure2RewardBlocks]

We noticed that assignment2 came with code that simplified the observation space by reducing the number of states in it. If the agent turned, even though the observation of blocks around the agent is the same, the code would alter the block positions to match how the agent turned, making it so the agent did not need to observe its current yaw. We were curious how important this simplification was, as it does not work as well once the agent is continuous, or at least, we suspect the agent may have a more accurate representation of the environment if it knows its exact yaw when using continuous commands. Therefore we tried the original prototype without the simplifier, instead providing the yaw as an additional value in the 1D array of blocks the agent was observing in the original prototype. This appears to have resulted in a similar rate of learning as the agent with the simplifier, and has a similar behavior of building an L shape structure at the start of the challenge. [Figure 3]

## Evaluation

## References
