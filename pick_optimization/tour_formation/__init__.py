"""
Tour Formation Module

This module provides functionality for forming pick tours from containers,
supporting different execution modes (local, distributed clustering/solving).
"""

# Primary entry point for containerized execution
from .tf_entry import run_tour_formation_entrypoint

# Core data structures and model definitions
from .tf_model import TourFormationModel 
from .tf_data import ModelData 
from .tf_solver import TourFormationResult 

# Specific components 
from .slack_calculator import SlackCalculator
from .clustering.clusterer import ContainerClusterer


# Define the public API of the module
__all__ = [
    # Entry point
    'run_tour_formation_entrypoint',

    # Core Structures
    'TourFormationModel',
    'ModelData',
    'TourFormationResult',
    'SlackCalculator',
    'ContainerClusterer'
] 