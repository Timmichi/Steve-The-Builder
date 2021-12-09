from enum import auto, Enum


class ProblemType(Enum):
    # Completely flat world, agent defends itself from a Ghast using Cobblestone.
    flat = auto()
    # Similar to flat world but involves a pyramid-like hill.
    hill = auto()
    # Agent attempts to prevent itself and sheep from falling into water
    # ponds scattered across the map.
    sheep_water = auto()