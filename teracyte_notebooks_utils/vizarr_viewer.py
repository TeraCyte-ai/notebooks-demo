"""
Vizarr viewer module for TeraCyte data overview notebooks.

This module provides interactive Vizarr viewer functionality for visualizing
FOV (Field of View) data stored in Azure blob storage as Zarr files.
"""

import traceback
import urllib.parse
from typing import Dict, Tuple, List

import ipywidgets as widgets
from IPython.display import HTML, IFrame, clear_output, display

from .sample import Sample
from .constants import MAGNIFICATION_GRID_CONFIG
from .api_utils import get_vizarr_url

def create_fovs_vizarr_viewer(sample: Sample) -> widgets.Widget:
    """
    Create a UI with a grid of FOV buttons in a snake pattern,
    automatically loading Vizarr when a button is clicked.
    
    Args:
        sample (Sample): Sample object containing all necessary metadata and paths
        
    Returns:
        widgets.Widget: Interactive UI widget for FOV selection and visualization
    """
    # Add custom CSS to the notebook first to style the iframe
    display(HTML("""<style>.jp-OutputArea-output iframe {
        min-width: 1000px !important;
        min-height: 700px !important;}</style>"""))
    
    # Output area for the iframe and messages
    output = widgets.Output()
    output.layout.height = '750px'  # Increased height for the viewer
    output.layout.width = '1000px'   # Increased width for the viewer
    output.layout.min_width = '1000px'
    output.layout.border = '1px solid #ddd'

    # Get grid configuration based on magnification
    magnification = sample.sample_metadata.get("magnification", 10)
    grid_config = MAGNIFICATION_GRID_CONFIG.get(magnification)
    if grid_config is None:
        raise ValueError(f"Unsupported magnification value: {magnification}")
    cols = grid_config["cols"]
    rows = grid_config["rows"]
    fov_count = grid_config["fov_count"]

    # Create mapping for FOV numbers to row/column coordinates using snake pattern
    fov_to_coords = _create_fov_mapping(rows, cols)

    def show_fov_in_vizarr(fov_num: int):
        """Display a specific FOV in Vizarr viewer."""
        with output:
            clear_output(wait=True)

            if fov_num not in fov_to_coords:
                print(f"Error: FOV number {fov_num} is not in the mapping")
                return

            row, col = fov_to_coords[fov_num]
            field = "0"  # Always using field 0

            path = f"{row}/{col}/{field}"
            print(f"Loading FOV #{fov_num} at {path}...")

            try:
                # Get the source URL using the sample object
                source_url = sample.get_fov_source_url(row, col, field)

                # Create the Vizarr URL with the source
                vizarr_url = (
                    f"{get_vizarr_url()}/"
                    f"?source={urllib.parse.quote(source_url)}"
                    f"&uiState=open&viewPosition=200,0?debug=true&t=123456789"
                )

                # Create an iframe to show Vizarr with increased size
                iframe = IFrame(
                    src=vizarr_url,
                    width="100%",
                    height=700  # Increased height for more space
                )
                display(iframe)

            except Exception as e:
                print(f"Error loading FOV: {str(e)}")
                traceback.print_exc()

    def show_all_fovs():
        """Display all FOVs together in Vizarr viewer."""
        with output:
            clear_output(wait=True)
            print("Loading all FOVs together...")

            try:
                # Get the source URL for the entire Zarr store
                source_url = sample.get_zarr_source_url()

                # Create the Vizarr URL with the source to view the entire plate
                vizarr_url = (
                    f"{get_vizarr_url()}/"
                    f"?source={urllib.parse.quote(source_url)}&uiState=open"
                )

                # Create an iframe to show Vizarr with increased size
                iframe = IFrame(
                    src=vizarr_url,
                    width="100%",
                    height=700
                )
                display(iframe)

            except Exception as e:
                print(f"Error loading all FOVs: {str(e)}")
                traceback.print_exc()

    def create_fov_button(fov_num: int) -> widgets.Button:
        """Create a button for a specific FOV number."""
        btn = widgets.Button(
            description=f"{fov_num}",
            layout=widgets.Layout(
                width='50px',
                height='40px',
                margin='2px',
                border='1px solid #ccc'
            )
        )

        def on_button_clicked(b):
            show_fov_in_vizarr(fov_num)

        btn.on_click(on_button_clicked)
        return btn

    # Create a grid with the buttons in a snake pattern
    grid_container = widgets.VBox([])
    grid_container.layout.margin = '0 25px 0 0'  # Add right margin
    grid_container.layout.min_width = '300px'

    # Create the visual grid for display (different from coordinate mapping)
    visual_grid = _create_visual_grid(rows, cols, fov_count)

    # Create the rows of buttons based on our pre-determined grid
    for row in visual_grid:
        row_buttons = widgets.HBox([])
        row_buttons.layout.justify_content = 'flex-start'
        row_buttons.layout.align_items = 'center'

        for fov_num in row:
            if fov_num < fov_count:  # Only create buttons for valid FOV numbers
                row_buttons.children += (create_fov_button(fov_num),)

        grid_container.children += (row_buttons,)

    # Create the "View All FOVs" button
    view_all_button = widgets.Button(
        description='View Chip',
        button_style='primary',
        layout=widgets.Layout(
            width='280px',
            height='50px',
            margin='20px 0 0 0',
            border='2px solid #007bff'
        )
    )

    view_all_button.on_click(lambda b: show_all_fovs())

    # Create the UI with the grid and output area side by side
    header = widgets.HTML("<h3>FOV Selection</h3>")
    header.layout.margin = '0 0 10px 0'

    # Add the view all button to the selection panel
    selector_panel = widgets.VBox([header, grid_container, view_all_button])
    selector_panel.layout.min_width = '300px'
    selector_panel.layout.margin = '10px'
    selector_panel.layout.align_items = 'flex-start'

    # Create a container that will position the grid and viewer side by side
    ui_container = widgets.HBox([selector_panel, output])
    ui_container.layout.align_items = 'flex-start'
    ui_container.layout.width = '100%'

    # Load the first FOV (0) by default
    show_fov_in_vizarr(0)

    return ui_container


def _create_fov_mapping(rows: int, cols: int) -> Dict[int, Tuple[str, str]]:
    """
    Create mapping from FOV numbers to row/column coordinates using snake pattern.
    
    Args:
        rows (int): Number of rows
        cols (int): Number of columns
        
    Returns:
        Dict[int, Tuple[str, str]]: Mapping from FOV number to (row_letter, col_num)
    """
    fov_to_coords = {}
    fov_num = 0
    
    for r in range(rows):
        if r % 2 == 0:  # Even rows (0, 2, 4) go left to right
            for c in range(cols):
                row_letter = chr(65 + r)  # Convert to A, B, C, etc.
                col_num = str(c + 1)      # Convert to 1, 2, 3, etc.
                fov_to_coords[fov_num] = (row_letter, col_num)
                fov_num += 1
        else:  # Odd rows (1, 3, 5) go right to left
            for c in range(cols - 1, -1, -1):
                row_letter = chr(65 + r)  # Convert to A, B, C, etc.
                col_num = str(c + 1)      # Convert to 1, 2, 3, etc.
                fov_to_coords[fov_num] = (row_letter, col_num)
                fov_num += 1
    
    return fov_to_coords


def _create_visual_grid(rows: int, cols: int, fov_count: int) -> List[List[int]]:
    """
    Create the visual grid layout for button display.
    
    Args:
        rows (int): Number of rows
        cols (int): Number of columns
        fov_count (int): Total number of FOVs
        
    Returns:
        List[List[int]]: 2D list representing the visual grid layout
    """
    if rows == 6 and cols == 5:  # 4x magnification
        return [
            [0, 1, 2, 3, 4],        # First row: left to right
            [9, 8, 7, 6, 5],        # Second row: right to left
            [10, 11, 12, 13, 14],   # Third row: left to right
            [19, 18, 17, 16, 15],   # Fourth row: right to left
            [20, 21, 22, 23, 24],   # Fifth row: left to right
            [29, 28, 27, 26, 25]    # Sixth row: right to left
        ]
    else:  # 10x magnification or custom
        visual_grid = []
        fov_num = 0
        
        for r in range(rows):
            row = []
            if r % 2 == 0:  # Even rows: left to right
                for c in range(cols):
                    if fov_num < fov_count:
                        row.append(fov_num)
                        fov_num += 1
            else:  # Odd rows: right to left
                temp_row = []
                for c in range(cols):
                    if fov_num < fov_count:
                        temp_row.append(fov_num)
                        fov_num += 1
                row = temp_row[::-1]  # Reverse for right to left
            
            if row:  # Only add non-empty rows
                visual_grid.append(row)
        
        return visual_grid
