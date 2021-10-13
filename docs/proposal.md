---
layout: default
title: Proposal
---

# Summary of Project
Problem: A common problem that arises for Minecraft players is surviving at nighttime. During nighttime, there are hostile animals, zombies, etc. that threaten the survival of our agent. Our agent should learn how to build some form of shelter before nighttime to survive against outside attacks.
Input: Use the grid surrounding the agent (3x3), and the existance/nonexistance of blocks in front of the agent (where the camera is directed).
Output: A shelter/structure that fully isolates the agent from its outside environment.
Applications: The agent would help players survive nighttime.

# AI/ML Algorithms
Reinforcement Learning using neural function approximation

# Evaluation Plan
To evaluate progress in our project, we will focus on how close to building a shelter the agent gets, determined by how "houselike" the blocks they have put down are. Some metrics will be how close blocks are to each other, whether a group of blocks in space constitute a "wall", how many wasted blocks the agent has placed, and how much time the agent is taking. Some ways to quantify these metrics include analyzing the environment near the agent to see if blocks are next to each other with no spaces in between, if more blocks than are necessary to build a shelter have been placed, and a decrementing timer for checking time taken.
Baseline: The shelter should cover/isolate our entire agent from the outside environment. (e.g. a big shelter with a lot of wasted blocks and time spent would be sufficient at first)
Some stepping stones to building a shelter may be the agent simply building a wall of one block height, then a wall of two block height, then two walls, then two walls perpendicular to each other. Since we are planning to use a neural network, it may be difficult to visualize the internals of our algorithm. We do plan to create some graphs to examine behavior of the agent over time, for example, how well they are optimizing the reward metrics. Our moonshot case would be the agent creating a shelter around itself pretty quickly. There are also some interesting alternative situations, like maybe the agent uses a nearby wall to build its house around so there's less work to do.

# Appointment with the Instructor
10/19/2021 @ 3:00pm
