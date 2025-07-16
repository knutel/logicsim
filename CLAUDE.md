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
2. **Dynamic UI Loading**: UI is loaded from external `.slint` files, allowing modifications without code changes
3. **Error Handling**: Comprehensive error handling with diagnostic output for debugging Slint files
4. **Modern Python**: Uses `pathlib`, proper error handling, and modern package management with `uv`

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

- **`logicsim/graph_data.py`**: Complete graph data structures with nodes, connectors, and connections
- **Node types**: Input nodes, logic gates (AND, OR, NOT), output nodes
- **Connector system**: Each node has precisely positioned connector points (black dots)
- **Connection lines**: Lines connect between specific connectors on nodes

### Graph Component Architecture
- **`ui/graph.slint`**: Dynamic graph rendering using data-driven components
- **NodeData struct**: Defines node properties (id, type, position, size, label)
- **ConnectionData struct**: Defines connection properties (id, start/end coordinates)
- **Dynamic rendering**: Uses `for` loops to render nodes and connections from data arrays

### Key Implementation Details

#### Slint Path Elements for Lines
- Connection lines use Slint's `Path` element with `MoveTo` and `LineTo` commands
- **Important**: Path coordinates are relative to the Path element bounds, not container
- Line positioning calculation: Path positioned at line start with width/height spanning to end
- Path coordinates: (0,0) to (width, height) for line endpoints

#### Data Structure Design
- **Graph definition**: Controlled entirely from Python side
- **Extensible**: Easy to add new node types and connection patterns
- **Precise positioning**: Mathematical calculation of connector positions for exact line connections
- **Data flow**: Python generates data → Slint renders components → Visual display

#### Debug Support
- **Slint debug()**: Use `debug()` function in Slint to verify data received from Python
- **Python logging**: Comprehensive logging with configurable levels
- **Diagnostic output**: Both Python and Slint sides provide detailed debug information

### Known Limitations
- **Python-to-Slint data binding**: Currently uses static data in Slint (Python data structure ready)
- **Connector positioning**: May need fine-tuning for pixel-perfect alignment
- **Future enhancement**: Full dynamic data binding from Python to Slint arrays

## Slint Interoperability Notes
- When assigining to a Slint property, Python lists must be wrapped in slint.ListModel