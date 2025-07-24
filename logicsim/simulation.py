"""
Circuit simulation engine for LogicSim

This module provides the core logic evaluation capabilities for simulating
digital logic circuits. It can evaluate individual nodes based on their
input values and node types.
"""

from typing import List
import logging

from .graph_data import Node

logger = logging.getLogger(__name__)


class CircuitEvaluator:
    """
    Evaluates logic gates and circuits for digital simulation.
    
    This class provides methods to evaluate individual nodes based on their
    type and input values. It supports standard logic gates (AND, OR, NOT)
    and can be extended to support additional gate types.
    """
    
    def __init__(self):
        """Initialize the circuit evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def _evaluate_and_gate(self, inputs: List[bool]) -> bool:
        """
        Evaluate an AND gate with the given inputs.
        
        Args:
            inputs: List of boolean input values
            
        Returns:
            bool: True if ALL inputs are True, False otherwise
            
        Raises:
            ValueError: If inputs list is empty or has wrong number of inputs
        """
        if len(inputs) != 2:
            raise ValueError(f"AND gate requires exactly 2 inputs, got {len(inputs)}")
        
        result = inputs[0] and inputs[1]
        self.logger.debug(f"AND gate: {inputs} -> {result}")
        return result
    
    def _evaluate_or_gate(self, inputs: List[bool]) -> bool:
        """
        Evaluate an OR gate with the given inputs.
        
        Args:
            inputs: List of boolean input values
            
        Returns:
            bool: True if ANY input is True, False otherwise
            
        Raises:
            ValueError: If inputs list is empty or has wrong number of inputs
        """
        if len(inputs) != 2:
            raise ValueError(f"OR gate requires exactly 2 inputs, got {len(inputs)}")
        
        result = inputs[0] or inputs[1]
        self.logger.debug(f"OR gate: {inputs} -> {result}")
        return result
    
    def _evaluate_not_gate(self, inputs: List[bool]) -> bool:
        """
        Evaluate a NOT gate with the given input.
        
        Args:
            inputs: List containing single boolean input value
            
        Returns:
            bool: Logical NOT of the input
            
        Raises:
            ValueError: If inputs list doesn't contain exactly one value
        """
        if len(inputs) != 1:
            raise ValueError(f"NOT gate requires exactly 1 input, got {len(inputs)}")
        
        result = not inputs[0]
        self.logger.debug(f"NOT gate: {inputs} -> {result}")
        return result
    
    def evaluate_node(self, node: Node, input_values: List[bool]) -> bool:
        """
        Evaluate a node based on its type and input values.
        
        Args:
            node: The node to evaluate
            input_values: List of boolean values for the node's inputs
            
        Returns:
            bool: The output value of the node
            
        Raises:
            ValueError: If node type is unsupported or input count is invalid
        """
        if node.node_type == "and":
            return self._evaluate_and_gate(input_values)
        elif node.node_type == "or":
            return self._evaluate_or_gate(input_values)
        elif node.node_type == "not":
            return self._evaluate_not_gate(input_values)
        elif node.node_type == "input":
            # Input nodes don't process inputs - they provide their own value
            if node.value is None:
                raise ValueError(f"Input node '{node.id}' has no value set")
            return node.value
        elif node.node_type == "output":
            # Output nodes pass through their input value
            if len(input_values) != 1:
                raise ValueError(f"Output node requires exactly 1 input, got {len(input_values)}")
            return input_values[0]
        else:
            raise ValueError(f"Unsupported node type: {node.node_type}")