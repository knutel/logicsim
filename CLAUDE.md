# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
uv run python main.py
```

### Package Management
```bash
uv sync                    # Install/sync dependencies
uv add package_name        # Add new dependency
uv add package_name --dev  # Add development dependency
```

### Development Workflow
- Use `uv run` to execute commands in the virtual environment
- The application loads UI from `ui/main.slint` dynamically
- Slint diagnostics are printed to help debug UI file issues

### Testing
```bash
uv run pytest                    # Run all tests (61 tests total)
uv run pytest tests/test_graph_data.py  # Run specific test file
uv run pytest -v                # Run with verbose output
```

### Debugging and Logging
```bash
LOGICSIM_LOG_LEVEL=DEBUG uv run python main.py  # Enable debug logging
LOGICSIM_LOG_LEVEL=INFO uv run python main.py   # Default logging level
```

## Architecture Overview

### Project Structure
- **Entry Point**: `main.py` - Application entry point with error handling and Slint integration
- **UI Definition**: `ui/main.slint` - Main window UI, `ui/graph.slint` - Graph visualization components
- **Graph Data**: `logicsim/graph_data.py` - Graph data structures and management
- **Package**: `logicsim/` - Python package structure
- **Dependencies**: Managed via `pyproject.toml` and `uv.lock`

### Key Architecture Patterns

1. **Separation of Concerns**: Python handles application logic, Slint handles UI definition
2. **Data-Driven Design**: Node types, connectors, and UI elements defined by data structures rather than hardcoded enums
3. **Dynamic UI Loading**: UI is loaded from external `.slint` files, allowing modifications without code changes
4. **Error Handling**: Comprehensive error handling with diagnostic output for debugging Slint files
5. **Modern Python**: Uses `pathlib`, proper error handling, and modern package management with `uv`

### Slint Integration
- UI components are defined in `.slint` files using declarative syntax
- Python loads Slint files using `slint.load_file()`
- Diagnostics are available for debugging UI compilation issues
- Standard widgets imported from `std-widgets.slint`

## Important Technical Details

### Python Version Requirements
- Requires Python >=3.10 (due to Slint dependency requirements)
- Uses pre-release versions of Slint (`slint>=1.0.0` with `--prerelease=allow`)

### UI Framework (Slint)
- Uses component-based architecture with `export component` syntax
- Layout uses containers like `VerticalBox` and `HorizontalBox`
- Properties and callbacks can be defined for component interaction
- Window properties (title, width, height) are set at component level

### Error Handling Pattern
The application includes comprehensive error handling that:
- Checks for UI file existence before loading
- Catches and reports Slint compilation errors
- Prints diagnostic information for debugging
- Provides user-friendly error messages

### Development Environment
- Uses `uv` for modern Python package management
- Virtual environment automatically managed by `uv`
- Claude Code permissions configured in `.claude/settings.local.json` for Slint documentation access

## Graph Visualization System

### Graph Data Management
The graph system uses Python for data management and Slint for rendering:

- **`logicsim/graph_data.py`**: Complete graph data structures with nodes, connectors, connections, and selection state
- **Data-driven architecture**: Node types defined by `NodeDefinition` and `ConnectorDefinition` classes, managed by `NodeDefinitionRegistry`
- **Node types**: Input nodes, logic gates (AND, OR, NOT), output nodes - all extensible through registry
- **Connector system**: Each node has precisely positioned connector points (black dots) calculated from ratio-based positioning
- **Connection lines**: Lines connect between specific connectors on nodes
- **Selection system**: Single node selection with mouse click handling and hit-testing

### Graph Component Architecture
- **`ui/graph.slint`**: Dynamic graph rendering using data-driven components with unified rectangle rendering
- **NodeData struct**: Defines node properties (id, type, position, size, label, color, connectors)
- **ConnectorData struct**: Defines connector properties (id, position, input/output type)
- **ConnectionData struct**: Defines connection properties (id, start/end coordinates)
- **Dynamic rendering**: Uses `for` loops to render nodes and connections from data arrays
- **Unified rendering**: All node types rendered as rectangles with color differentiation

### Key Implementation Details

#### Slint Path Elements for Lines
- Connection lines use Slint's `Path` element with `MoveTo` and `LineTo` commands
- **Important**: Path coordinates are relative to the Path element bounds, not container
- Line positioning calculation: Path positioned at line start with width/height spanning to end
- Path coordinates: (0,0) to (width, height) for line endpoints
- **Memory**: We need to set viewbox-height and viewbox-width of the Path element. Otherwise the viewbox will be set to the extents of the contents of the Path element.

#### Data Structure Design
- **Graph definition**: Controlled entirely from Python side with registry pattern for extensibility
- **Node definitions**: `NodeDefinition` class with connector templates using ratio-based positioning (0.0-1.0)
- **Connector positioning**: Calculated from ratios to support dimension-independent scaling
- **Registry pattern**: `NodeDefinitionRegistry` for managing and extending available node types
- **Selection state**: Tracked in `GraphData` class with `selected_node_id` field
- **Data flow**: Python generates data → Slint renders components → Visual display
- **Slint format**: `to_slint_format()` includes nodes, connections, and selected_nodes arrays

#### Debug Support
- **Slint debug()**: Use `debug()` function in Slint to verify data received from Python
- **Python logging**: Comprehensive logging with configurable levels
- **Diagnostic output**: Both Python and Slint sides provide detailed debug information

#### Mouse Interaction and Selection
- **Hit-testing**: `is_point_in_node()` checks if mouse coordinates are within node boundaries
- **Node selection**: `handle_mouse_click()` manages single node selection with proper state transitions
- **Selection behavior**: Click to select, click same node to deselect, click different node to switch selection
- **Empty area clicks**: Clicking outside nodes clears current selection
- **Overlapping nodes**: Later-added nodes take priority for selection (drawing order)

### Known Limitations and Future Enhancements
- **UI mouse events**: Python selection methods ready, but UI click event handling not yet connected to Slint
- **Connector positioning**: Pixel-perfect alignment achieved with ratio-based calculations
- **Visual selection feedback**: Selected nodes not yet visually highlighted in UI (data structure ready)

## Slint Interoperability Notes
- When assigning to a Slint property, Python lists must be wrapped in slint.ListModel
- **Color format**: Slint expects rgb() format colors, not hex strings
- **Data conversion**: Python data structures must be converted to Slint-compatible types (str, float, bool)
- **Struct compatibility**: Python dictionaries map directly to Slint struct definitions

## Testing Strategy
- **Comprehensive coverage**: 61 unit tests covering all core functionality
- **Parameterized tests**: Used for testing multiple node types with same logic patterns
- **Mock logging**: Logger behavior tested with unittest.mock.patch
- **Selection testing**: Complete test coverage for all mouse interaction scenarios
- **Data structure validation**: Tests verify correct connector positioning calculations
- **Error handling**: Tests cover invalid inputs and edge cases