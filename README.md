# LogicSim - Logic Circuit Simulator

A modern, interactive logic circuit simulator built with Python and Slint UI, featuring advanced net routing with waypoints and real-time simulation capabilities.

![LogicSim Screenshot](Screenshot%202025-07-25%20224113.png)

## Features

### 🎯 Interactive Circuit Design
- **Drag-and-drop interface** with toolbox containing logic gates (AND, OR, NOT, NOR)
- **Input and output nodes** for circuit interfacing
- **Visual node connector points** for precise wire connections
- **Real-time visual feedback** during circuit construction

### 🔌 Advanced Net Routing System
- **Multi-node nets** supporting complex connection patterns (fan-out, wired-OR)
- **Waypoint-based routing** for sophisticated wire paths:
  - Double-click net segments to add waypoints
  - Drag waypoints to customize wire routing
  - Connect nets from/to waypoints for complex topologies
- **Sequential waypoint chains** (no star topology limitations)
- **Visual line segments** with proper graph-based rendering

### ⚡ Circuit Simulation
- **Real-time simulation** with visual state representation
- **Node value visualization** (green=high, red=low, gray=undefined)
- **Net value propagation** with color-coded wires
- **Sequential circuit support** with feedback loop detection
- **Iterative simulation** for SR latches and other memory elements

### 🎮 User Interface
- **Dual-mode operation**: Edit mode and Simulation mode
- **Node selection and editing** with double-click label editing
- **Wire selection and deletion** for easy circuit modification
- **Toolbox integration** with visual node type selection
- **Keyboard shortcuts and mouse interactions**

## Getting Started

### Prerequisites
- Python ≥3.10
- uv package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd logicsim
```

2. Install dependencies:
```bash
uv sync
```

3. Run the application:
```bash
uv run python main.py
```

## Usage

### Basic Circuit Construction

1. **Adding Components**: Click node types in the toolbox, then click in the canvas to place them
2. **Connecting Components**: Click a connector (black dot) on one node, then click a connector on another node
3. **Adding Waypoints**: Double-click any wire segment to add a routing waypoint
4. **Advanced Routing**: Double-click waypoints to start new connections, or connect directly to waypoints

### Simulation

1. **Enter Simulation Mode**: Click "Enter Simulation" button
2. **Set Input Values**: Double-click input nodes to toggle their values (true/false)
3. **Observe Results**: Watch the circuit simulate in real-time with colored visual feedback
4. **Return to Editing**: Click "Enter Edit" to modify the circuit

### Example Circuits

The application includes a pre-loaded **SR NOR Latch** demonstration showing:
- Input nodes for Set and Reset signals
- Cross-coupled NOR gates creating memory behavior  
- Output nodes showing Q and Q̄ (complementary outputs)
- Complex wire routing with waypoints

## Architecture

### Core Components

- **Python Backend** (`logicsim/graph_data.py`): Graph data structures, simulation engine, waypoint topology management
- **Slint UI** (`ui/*.slint`): Declarative UI components for rendering and interaction
- **Main Application** (`main.py`): Application entry point with comprehensive error handling

### Key Features

- **Data-driven architecture**: Node types and UI elements defined by data structures
- **Graph-based simulation**: Topological sorting with feedback loop detection  
- **Modern Python practices**: Type hints, dataclasses, pathlib, proper error handling
- **Comprehensive testing**: 289+ unit tests covering all functionality

## Development

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_graph_data.py
```

### Debug Logging
```bash
# Enable debug logging
LOGICSIM_LOG_LEVEL=DEBUG uv run python main.py

# Default info level
LOGICSIM_LOG_LEVEL=INFO uv run python main.py
```

### Project Structure
```
logicsim/
├── logicsim/           # Python package
│   └── graph_data.py   # Core graph data structures and simulation
├── ui/                 # Slint UI components
│   ├── main.slint      # Main window definition
│   ├── graph.slint     # Graph visualization components
│   └── toolbox.slint   # Node type toolbox
├── tests/              # Comprehensive test suite
├── main.py             # Application entry point
└── pyproject.toml      # Project configuration
```

## Technical Highlights

### Waypoint System
- **Sequential topology**: Waypoints form chains rather than star connections
- **Graph-based routing**: Proper segment calculation using connection topology
- **Visual integration**: Seamless rendering with multi-segment Path elements
- **Interactive editing**: Full support for waypoint creation, deletion, and dragging

### Simulation Engine
- **Multi-algorithm approach**: Combinational (topological sort) and sequential (iterative) simulation
- **Feedback detection**: Tarjan's SCC algorithm for cycle detection
- **Value propagation**: Efficient net-based value distribution
- **State management**: Proper handling of undefined and conflicting values

### Modern UI Framework
- **Slint integration**: Native performance with declarative syntax
- **Component architecture**: Reusable UI components with clean separation
- **Event handling**: Comprehensive mouse and keyboard interaction support
- **Dynamic rendering**: Data-driven UI updates with efficient re-rendering

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow existing code style and patterns
- Add comprehensive tests for new functionality
- Update documentation for user-facing changes
- Use descriptive commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Slint UI Framework** for providing excellent Python integration
- **uv Package Manager** for modern Python dependency management
- **pytest** for comprehensive testing capabilities

---

**LogicSim** - Where logic meets intuitive design! 🔬⚡