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

## Architecture Overview

### Project Structure
- **Entry Point**: `main.py` - Application entry point with error handling and Slint integration
- **UI Definition**: `ui/main.slint` - Slint UI files for interface definition
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