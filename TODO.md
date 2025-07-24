# Circuit Simulation Implementation Plan

This document outlines the step-by-step implementation of circuit simulation capabilities for the LogicSim graph editor. Each step is designed to keep the application in a fully functional state while progressively adding simulation features.

## Step 1: Add Basic Node State Data Structure
**Goal**: Extend the data model to support node states for simulation

### Tasks:
- Add `value: Optional[bool]` field to `Node` class in `logicsim/graph_data.py`
- Update `Node.__init__()` to initialize the new field
- Add node state to `to_slint_format()` method for UI consumption
- Update all existing tests to handle the new field
- Add new tests for node state functionality

### Expected Outcome:
- Graph data can store and track boolean states for each node
- UI data format includes node state information
- All existing functionality remains intact
- Foundation for simulation logic is established

## Step 2: Create Minimal Logic Evaluation Engine
**Goal**: Build core simulation logic for evaluating individual nodes

### Tasks:
- Create new file `logicsim/simulation.py`
- Implement `CircuitEvaluator` class with basic structure
- Add logic gate evaluation methods:
  - `_evaluate_and_gate(inputs: List[bool]) -> bool`
  - `_evaluate_or_gate(inputs: List[bool]) -> bool`
  - `_evaluate_not_gate(inputs: List[bool]) -> bool`
- Add `evaluate_node(node: Node, input_values: List[bool]) -> bool` method
- Create comprehensive unit tests for all logic functions

### Expected Outcome:
- Core simulation engine exists and can be tested independently
- Logic gate behavior is implemented and verified
- Foundation for circuit-wide evaluation is ready

## Step 3: Add Input Node Value Setting
**Goal**: Enable programmatic control of input node values

### Tasks:
- Add `set_input_value(node_id: str, value: bool)` method to `GraphData`
- Add input validation to ensure only input nodes can be set
- Add `get_node_value(node_id: str) -> Optional[bool]` method
- Update `to_slint_format()` to include input node states
- Add tests for input value setting and validation

### Expected Outcome:
- Input nodes can be controlled programmatically
- Proper validation prevents invalid operations
- UI can display input node states
- Ready for user interaction implementation

## Step 4: Add Visual State Indicators
**Goal**: Provide visual feedback for node states in the UI

### Tasks:
- Update `NodeData` struct in `ui/graph.slint` to include state field
- Modify node rendering to show different colors based on state:
  - Gray: undefined/not evaluated
  - Green: true/high
  - Red: false/low
- Update node rendering logic to use state-based colors
- Test visual indicators with sample data

### Expected Outcome:
- Nodes visually indicate their current state
- Color coding provides immediate feedback
- UI becomes more engaging and informative
- Ready for interactive simulation

## Step 5: Add Circuit Evaluation Integration
**Goal**: Connect simulation engine to graph data for full circuit evaluation

### Tasks:
- Add `simulate()` method to `GraphData` class
- Implement topological sorting for proper evaluation order
- Add dependency resolution for node evaluation
- Handle circular dependencies and invalid circuits gracefully
- Add comprehensive tests for circuit evaluation scenarios

### Expected Outcome:
- Complete circuits can be evaluated automatically
- Proper evaluation order ensures correct results
- Error handling for invalid circuit configurations
- Full simulation capability is functional

## Step 6: Add Basic UI Interaction
**Goal**: Enable user interaction for running simulations

### Tasks:
- Connect "Simulate" button to trigger circuit evaluation
- Add click handlers for input nodes to toggle their values
- Update UI state after simulation runs
- Add user feedback for simulation errors
- Test complete user workflow

### Expected Outcome:
- Users can interactively run simulations
- Input values can be toggled by clicking nodes
- Visual feedback shows simulation results immediately
- Complete circuit simulation workflow is functional

## Implementation Notes

### Code Quality Standards:
- All new code must include comprehensive unit tests
- Follow existing code style and patterns
- Add proper error handling and logging
- Document public APIs with docstrings

### Testing Strategy:
- Unit tests for all new classes and methods
- Integration tests for complete workflows
- Edge case testing for invalid inputs
- Performance testing for complex circuits

### Future Enhancements (Not in Current Plan):
- Real-time simulation with animated signal propagation
- Additional gate types (XOR, NAND, NOR, etc.)
- Multi-bit signals and buses
- Timing simulation and delay modeling
- Circuit analysis tools (critical path, power consumption)

## Success Criteria

Each step is considered complete when:
1. All planned functionality is implemented
2. Comprehensive tests pass
3. Application runs without errors
4. Existing functionality remains unchanged
5. Code review standards are met

The final result will be a fully interactive circuit simulator built incrementally on the solid foundation of the existing graph editor.