---
layout: default
title: Status
---
## Project Summary
Steve the Builder is a project focused on exploring the ways machine learning can be used to control a bot in Minecraft whose main defense against opposing creatures is simply building. The goal is to have the bot create a shelter for itself against different types of enemies and in different kinds of terrains.

## Approach
Since the first enemy we tested, the "Ghast", only shoots a harmful projectile to the agent every 4~6 seconds, it was evident that some kind of delayed reinforcement learning algorithm would be appropriate.

related notes:
    - the agent can only look around (up, down, left, right) and place blocks. Cannot move because don't want to encourage the agent to just start dodging enemies. May simply give a negative reward for this in the future.
## Evaluation
time reward increase over time
qualitative: seeing agent build to avoid enemies
## Remaining Goals and Challenges

## Resources Used
assignment2 was used as the base or starting point for this project.

For examples on how the documentation is converted to code:
https://github.com/Microsoft/malmo/tree/master/Malmo/samples/Python_examples

Learning what commands can be inputted to Malmo:
https://microsoft.github.io/malmo/0.30.0/Schemas/MissionHandlers.html#type_DiscreteMovementCommand

Malmo documentation:
https://microsoft.github.io/malmo/0.30.0/Schemas/Mission.html

A map of specific features to the official Python examples that contain them:
https://canvas.eee.uci.edu/courses/34142/pages/python-examples-malmo-functionality

Finding and quickly comparing different off-the-shelf algorithms:
https://docs.ray.io/en/latest/rllib-algorithms.html