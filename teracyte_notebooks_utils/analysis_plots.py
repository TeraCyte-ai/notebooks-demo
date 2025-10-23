# Standard library
import json
from uuid import uuid4
from itertools import islice
from IPython.display import display as ipython_display

# Scientific / data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN

# Interactive widgets & display
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Text, Dropdown, ToggleButtons, SelectMultiple
from IPython.display import display, clear_output, HTML

# Plotly
import plotly.graph_objects as go

# Bokeh
from bokeh.io import output_notebook
output_notebook()
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import push_notebook
from bokeh.embed import json_item
from bokeh.layouts import column
from bokeh.transform import transform, factor_cmap
from bokeh.models import (
    ColumnDataSource,
    LinearColorMapper,
    LogColorMapper,
    ColorBar,
    Range1d,
    TextAreaInput,
    CustomJS,
    HoverTool,
    PanTool,
    WheelZoomTool,
    BoxZoomTool,
    ResetTool,
    SaveTool,
)
from bokeh.models.scales import LinearScale, LogScale
from bokeh.palettes import Category10, Category20, Turbo256, Viridis256

# Project-specific
from .constants import PARQUET_FILES, MAGNIFICATION_GRID_CONFIG
from .vizarr_viewer import _create_visual_grid
from .sample import Sample
from .data_query import get_assay_workflows_status


def parquet_seq_fov_heatmaps(sample: Sample, parquet_data_progress: pd.DataFrame):
    magnification = sample.sample_metadata.get("magnification", 10)
    grid_config = MAGNIFICATION_GRID_CONFIG.get(magnification)
    if grid_config is None:
        raise ValueError(f"Unsupported magnification value: {magnification}")
    cols = grid_config["cols"]
    rows = grid_config["rows"]
    fov_count = grid_config["fov_count"]

    visual_grid = _create_visual_grid(rows, cols, fov_count)
    visual_grid_array = np.array(visual_grid)
    
    for seq in parquet_data_progress.itertuples():
        seq_vis_progress = np.ones_like(visual_grid_array, dtype=int)
        seq_vis_progress[np.isin(visual_grid_array, seq.missing_fovs)] = 0
        seq_vis_progress[np.isin(visual_grid_array, seq.fovs)] = 1  # optional; already 1

        cmap = ListedColormap(["#ff9999", "#99ff99"])
        _, ax = plt.subplots()
        _ = ax.imshow(seq_vis_progress, cmap=cmap, vmin=0, vmax=1)

        for i in range(visual_grid_array.shape[0]):
            for j in range(visual_grid_array.shape[1]):
                ax.text(j, i, visual_grid_array[i, j],
                        ha="center", va="center", color="black", fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(f"Parquet data in sequence {seq.sequence}")
        plt.show()


def workflows_progress_heatmaps(sample: Sample, workflows_df: pd.DataFrame):
    """
    Create heatmaps showing workflow progress for each sequence using matplotlib.
    
    Color scheme:
    - Red: Failed
    - Green: Succeeded  
    - Blue: Running
    - Yellow: Pending
    - Grey: Not exist in table
    """    
    magnification = sample.sample_metadata.get("magnification", 10)
    grid_config = MAGNIFICATION_GRID_CONFIG.get(magnification)
    if grid_config is None:
        raise ValueError(f"Unsupported magnification value: {magnification}")
    
    cols = grid_config["cols"]
    rows = grid_config["rows"]
    fov_count = grid_config["fov_count"]

    visual_grid = _create_visual_grid(rows, cols, fov_count)
    visual_grid_array = np.array(visual_grid)
    
    # Define color mapping - RGB values for matplotlib
    phase_colors = {
        'Failed': '#FFB3BA',      # Pastel Red
        'Succeeded': '#BAFFC9',   # Pastel Green  
        'Running': '#BAE1FF',     # Pastel Blue
        'Pending': '#FFFFBA',     # Pastel Yellow
        'Not exist': '#D3D3D3'    # Light Grey
    }
    
    # Get unique sequences
    sequences = sorted(workflows_df['seq_num'].unique()) if 'seq_num' in workflows_df.columns else [0]
    
    for seq in sequences:
        # Filter workflows for this sequence
        seq_workflows = workflows_df[workflows_df['seq_num'] == seq] if 'seq_num' in workflows_df.columns else workflows_df
        
        # Create color matrix
        color_matrix = np.full(visual_grid_array.shape, 'Not exist', dtype=object)
        
        # Create lookup dictionaries for faster access
        fov_to_workflow = {}
        for _, workflow in seq_workflows.iterrows():
            fov_num = workflow['fov_num']
            fov_to_workflow[fov_num] = workflow
        
        # Fill the color matrix
        for i in range(visual_grid_array.shape[0]):
            for j in range(visual_grid_array.shape[1]):
                fov_num = int(visual_grid_array[i, j])
                
                if fov_num in fov_to_workflow:
                    workflow = fov_to_workflow[fov_num]
                    phase = workflow['phase']
                    color_matrix[i, j] = phase
                else:
                    color_matrix[i, j] = 'Not exist'
        
        # Create the matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a custom colormap from our phase colors
        from matplotlib.colors import ListedColormap
        import matplotlib.patches as mpatches
        
        # Create color array for the heatmap
        numeric_matrix = np.zeros(visual_grid_array.shape)
        color_list = []
        phase_to_num = {}
        
        unique_phases = list(set(color_matrix.flatten()))
        for idx, phase in enumerate(unique_phases):
            phase_to_num[phase] = idx
            color_list.append(phase_colors[phase])
        
        # Convert phase matrix to numeric matrix
        for i in range(color_matrix.shape[0]):
            for j in range(color_matrix.shape[1]):
                numeric_matrix[i, j] = phase_to_num[color_matrix[i, j]]
        
        # Create custom colormap
        cmap = ListedColormap(color_list)
        
        # Create the heatmap
        im = ax.imshow(numeric_matrix, cmap=cmap, aspect='equal', 
                      vmin=0, vmax=len(unique_phases)-1)
        
        # Add FOV numbers as text
        for i in range(visual_grid_array.shape[0]):
            for j in range(visual_grid_array.shape[1]):
                fov_num = visual_grid_array[i, j]
                ax.text(j, i, str(fov_num), ha='center', va='center', 
                       fontsize=10, color='black', weight='normal')
        
        # Create legend with all possible phases (not just the ones in current data)
        all_phases = ['Succeeded', 'Running', 'Pending', 'Failed', 'Not exist']
        legend_elements = []
        for phase in all_phases:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=phase_colors[phase], 
                                            markersize=10, label=phase))
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Set title and remove axes
        workflow_name = seq_workflows.iloc[0]['workflow_name'] if len(seq_workflows) > 0 else 'No Data'
        ax.set_title(f"Workflow Progress - {workflow_name} - Sequence {seq}", 
                    fontsize=14, pad=20)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Print workflow summary
        if len(seq_workflows) > 0:
            phase_counts = seq_workflows['phase'].value_counts()
            print(f"\n📊 Workflow Summary for Sequence {seq}: Total {len(seq_workflows)}")
            for phase, count in phase_counts.items():
                print(f"  {phase}: {count}")


def create_workflow_selector(sample: Sample):
    """
    Create interactive dropdown widgets for selecting sequences and workflow names
    """    
    # Get workflow status data
    workflows_df = get_assay_workflows_status(sample.assay_id)
    
    # Get unique sequences and workflow names
    sequences = sorted(workflows_df['seq_num'].unique()) if 'seq_num' in workflows_df.columns else [0]
    workflow_names = sorted(workflows_df['workflow_name'].unique()) if 'workflow_name' in workflows_df.columns else ['default']
    
    # No 'All' option - use actual values only
    sequences_options = [str(seq) for seq in sequences]
    workflow_options = list(workflow_names)
    
    # Create dropdown widgets
    sequence_dropdown = widgets.Dropdown(
        options=sequences_options,
        value=sequences_options[0] if sequences_options else '0',
        description='Sequence:',
        disabled=False,
        style={'description_width': 'initial'}
    )
    
    workflow_dropdown = widgets.Dropdown(
        options=workflow_options,
        value=workflow_options[0] if workflow_options else workflow_names[0],
        description='Workflow:',
        disabled=False,
        style={'description_width': 'initial'}
    )
    
    # Create output widget for the heatmap
    output = widgets.Output()
    
    def update_heatmap(*args):
        """Update heatmap based on selected filters"""
        with output:
            clear_output(wait=True)
            
            # Filter dataframe based on selections
            filtered_df = workflows_df.copy()
            
            # Filter by sequence (always filter, no 'All' option)
            if 'seq_num' in filtered_df.columns:
                selected_seq = int(sequence_dropdown.value)
                filtered_df = filtered_df[filtered_df['seq_num'] == selected_seq]
            
            # Filter by workflow name (always filter, no 'All' option)
            if 'workflow_name' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['workflow_name'] == workflow_dropdown.value]
            
            # Show summary of filtered data
            if len(filtered_df) > 0:
                workflows_progress_heatmaps(sample, filtered_df)
            else:
                print("❌ No workflows found matching the selected criteria.")
    
    # Set up event handlers
    sequence_dropdown.observe(update_heatmap, names='value')
    workflow_dropdown.observe(update_heatmap, names='value')
    
    # Initial display
    update_heatmap()
    
    # Display the interface
    controls = widgets.HBox([sequence_dropdown, workflow_dropdown])
    interface = widgets.VBox([
        widgets.HTML("<h3>🎛️ Workflow Progress Selector</h3>"),
        controls,
        output
    ])
    
    return interface


def create_dataframe_selector(sample):
    """
    Create a widget to select which dataframe to work with.
    Options: 'cells', 'wells', 'colocalization'
    Returns the widget.
    """
    df_options = sample.available_parquet_types
    
    selector = widgets.Dropdown(
        options=df_options,
        value=df_options[0] if len(df_options) > 0 else None,
        description="Select DataFrame:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    display(selector)
    return selector


def remove_outliers(series):
    series = pd.to_numeric(series, errors='coerce').dropna()  
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]


def remove_outliers_from_dataframe(df, feature_columns=None, groupby_columns=['sequence', 'channel_index', 'fov', 'z_index'], data_type="wells_data"):
    """
    Remove outliers from specified feature columns in a DataFrame using IQR method.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        feature_columns (list): List of feature columns to process. If None, uses PARQUET_FILES[data_type]["feature_columns"]
        groupby_columns (list): Columns to group by when calculating outliers
        data_type (str): The data type key to use for PARQUET_FILES lookup
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed
        
    Raises:
        ValueError: If inputs are invalid
    """

    try:
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
        
        if df.empty:
            print("⚠️ Warning: DataFrame is empty, returning unchanged")
            return df.copy()
        
        if feature_columns is None:
            feature_columns = PARQUET_FILES[data_type]["feature_columns"]
        
        # Filter to only include features that exist in the DataFrame
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not isinstance(feature_columns, (list, tuple)):
            raise TypeError(f"feature_columns must be a list or tuple, got {type(feature_columns).__name__}")
        
        if not isinstance(groupby_columns, (list, tuple)):
            raise TypeError(f"groupby_columns must be a list or tuple, got {type(groupby_columns).__name__}")
        
        # Check if groupby columns exist
        missing_groupby_cols = [col for col in groupby_columns if col not in df.columns]
        if missing_groupby_cols:
            print(f"⚠️ Warning: Missing groupby columns {missing_groupby_cols}, skipping grouping")
            groupby_columns = []
        
        result_df = df.copy()
        processed_features = 0
        
        print(f"🧹 Starting outlier removal for {len(available_features)} features:")
        
        for feature in available_features:
            try:
                original_count = result_df[feature].notna().sum()
                
                if original_count == 0:
                    print(f"⚠️ Warning: No valid data for feature '{feature}', skipping")
                    continue
                
                # Apply outlier removal
                if groupby_columns:
                    filtered_values = (
                        result_df.groupby(groupby_columns)[feature]
                        .transform(remove_outliers)
                    )
                else:
                    # Apply outlier removal to entire column if no grouping
                    filtered_values = remove_outliers(result_df[feature])
                
                # Handle dtype compatibility when assigning back
                if filtered_values.dtype != result_df[feature].dtype:
                    # Convert to nullable integer type if original was integer
                    if pd.api.types.is_integer_dtype(result_df[feature].dtype):
                        result_df[feature] = filtered_values.astype('Int64')
                    else:
                        result_df[feature] = filtered_values
                else:
                    result_df.loc[:, feature] = filtered_values
                
                final_count = result_df[feature].notna().sum()
                removed_count = original_count - final_count
                removal_pct = (removed_count / original_count * 100) if original_count > 0 else 0
                
                print(f"   ✅ {feature}: {removed_count} outliers removed ({removal_pct:.1f}%) - {final_count}/{original_count} values retained")
                processed_features += 1
                
            except Exception as e:
                print(f"❌ Error processing feature '{feature}': {str(e)}")
                continue
        
        print(f"🎯 Outlier removal completed: {processed_features}/{len(available_features)} features processed")
        return result_df
        
    except Exception as e:
        print(f"❌ Critical error in remove_outliers_from_dataframe: {str(e)}")
        import traceback
        traceback.print_exc()
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def create_outlier_filtering_controls(data, data_type="wells_data"):
    """Create interactive controls for outlier filtering option with advanced parameters."""
    try:
        # Initialize filtered_df with original data (will be updated when button is clicked)
        global filtered_df
        filtered_df = data.copy()
        
        # Get available feature columns and groupby columns
        available_features = [col for col in PARQUET_FILES[data_type]["feature_columns"] if col in data.columns]
        available_groupby_cols = ['None', 'sequence', 'channel_index', 'fov', 'z_index']
        # Filter available columns (but keep 'None' as a special option)
        data_columns = [col for col in available_groupby_cols[1:] if col in data.columns]  # Skip 'None'
        available_groupby_cols = ['None'] + data_columns
        
        # Create widgets
        filter_outliers_widget = widgets.Checkbox(
            value=True,  # Default to applying outlier removal
            description='Apply Outlier Filtering',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Feature columns selection
        feature_columns_widget = widgets.SelectMultiple(
            options=available_features,
            value=available_features,  # Select all by default
            description='Feature Columns:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='150px')
        )
        
        # Groupby columns selection - set default to available columns (excluding 'None')
        default_groupby = [col for col in ['sequence', 'channel_index', 'fov','z_index'] if col in data_columns]
        groupby_columns_widget = widgets.SelectMultiple(
            options=available_groupby_cols,
            value=default_groupby,  # Default grouping with available columns
            description='Group By Columns:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px', height='120px')
        )
        
        # Advanced options toggle
        show_advanced_widget = widgets.Checkbox(
            value=False,
            description='Show Advanced Options',
            style={'description_width': 'initial'}
        )
        
        # Container for advanced options
        advanced_options = widgets.VBox([
            widgets.HTML("<b>Advanced Outlier Filtering Options:</b>"),
            widgets.HTML("<p style='color: #666; font-size: 12px;'>Select specific features and grouping strategy for outlier detection:</p>"),
            feature_columns_widget,
            groupby_columns_widget,
            widgets.HTML("<p style='color: #666; font-size: 11px;'><b>Group By:</b> Outliers calculated within each group combination. Select 'None' for global outlier removal across all data.</p>")
        ], layout=widgets.Layout(display='none'))
        
        def toggle_advanced_options(change):
            advanced_options.layout.display = 'block' if change['new'] else 'none'
        
        show_advanced_widget.observe(toggle_advanced_options, names='value')
        
        info_widget = widgets.HTML(
            value="<p style='color: #666; font-size: 14px; margin: 10px 0;'>"
                  "Enable outlier filtering to remove extreme values using the IQR method. "
                  "Disable if you want to preserve all data points for analysis.</p>"
        )
        
        apply_button = widgets.Button(
            description='⚗️ Apply Filter Settings',
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        output_area = widgets.Output()
        
        def on_apply_click(button):
            """Apply the filtering choice and create filtered_df."""
            with output_area:
                output_area.clear_output()
                
                apply_filtering = filter_outliers_widget.value
                
                if apply_filtering:
                    print("🔧 Applying outlier removal to the data...")
                    
                    # Get parameters from controls
                    selected_features = list(feature_columns_widget.value) if show_advanced_widget.value else None
                    raw_groupby = list(groupby_columns_widget.value) if show_advanced_widget.value else default_groupby
                    
                    # Handle 'None' option for global outlier removal
                    if 'None' in raw_groupby:
                        selected_groupby = []  # Empty list means global outlier removal
                    else:
                        selected_groupby = raw_groupby
                    
                    # Display configuration
                    print(f"📋 Configuration:")
                    print(f"   Feature columns: {len(selected_features) if selected_features else 'All available'} features")
                    if not selected_groupby:
                        print(f"   Group by: Global (no grouping)")
                    else:
                        print(f"   Group by: {selected_groupby}")
                    
                    try:
                        global filtered_df
                        filtered_df = remove_outliers_from_dataframe(
                            data, 
                            feature_columns=selected_features,
                            groupby_columns=selected_groupby,
                            data_type=data_type
                        )
                        
                    except Exception as e:
                        print(f"❌ Error applying outlier filtering: {str(e)}")
                        print("📋 Using original data without filtering...")
                        filtered_df = data.copy()
                else:
                    print("📋 Skipping outlier filtering - using original data...")
                    filtered_df = data.copy()
                    print(f"✅ Using all {len(filtered_df):,} data points for analysis")
                
        
        apply_button.on_click(on_apply_click)
        
        # Create the controls layout
        controls_layout = widgets.VBox([
            widgets.HTML("<h4>🔍 Outlier Filtering Options</h4>"),
            info_widget,
            filter_outliers_widget,
            show_advanced_widget,
            advanced_options,
            apply_button,
            output_area
        ])
        
        display(controls_layout)
        
        # Store references for external access
        controls_layout.filter_outliers_widget = filter_outliers_widget
        controls_layout.feature_columns_widget = feature_columns_widget
        controls_layout.groupby_columns_widget = groupby_columns_widget
        controls_layout.show_advanced_widget = show_advanced_widget
        controls_layout.output_area = output_area
        
        return filtered_df
        
    except Exception as e:
        print(f"❌ Error creating outlier filtering controls: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_chip_layout(Objective):
    """
    Generate a 2D chip layout with a serpentine (zig-zag) scan pattern.

    Parameters:
        Objective (int): Microscope objective magnification. Must be 4 or 10.

    Returns:
        np.ndarray: 2D layout array with FOV indices.
        
    Raises:
        ValueError: If Objective is not supported or invalid.
        TypeError: If Objective is not a numeric type.
    TODO Add also 1X objective
    """
    try:
        # Input validation
        if not isinstance(Objective, (int, float)):
            raise TypeError(f"Objective must be a number, got {type(Objective).__name__}")
        
        Objective = int(Objective)  # Convert to int 
        
        if Objective == 4:
            fov_count = 30
            columns = 5
            rows = 6
        elif Objective == 10:
            fov_count = 154
            columns = 11
            rows = 14
        else:
            raise ValueError(f"Unsupported Objective value: {Objective}. Supported values are 4 and 10.")

        # Validate calculated parameters
        if rows <= 0 or columns <= 0 or fov_count <= 0:
            raise ValueError(f"Invalid layout parameters: rows={rows}, columns={columns}, fov_count={fov_count}")

        layout = np.full((rows, columns), -1, dtype=int)  # Initialize with -1 for clarity
        fov = 0

        for row in range(rows):
            try:
                if fov + columns > fov_count:
                    actual_columns = fov_count - fov  # in case the last row is incomplete
                else:
                    actual_columns = columns

                # Ensure we don't go out of bounds
                if actual_columns <= 0:
                    break
                    
                if row % 2 == 0:
                    layout[row, :actual_columns] = range(fov, fov + actual_columns)
                else:
                    layout[row, :actual_columns] = range(fov + actual_columns - 1, fov - 1, -1)

                fov += actual_columns
                
            except Exception as e:
                print(f"⚠️ Warning: Error processing row {row}: {str(e)}")
                continue

        return layout
        
    except Exception as e:
        print(f"❌ Error generating chip layout: {str(e)}")
        raise


def map_fov_to_chip(df, layout, parameter):
    """
    Map values from the dataframe onto the 2D chip layout.

    - This fills the chip layout with the value of a specific parameter from each FOV
    - If a FOV is missing from the data, it is left as NaN.

    Parameters:
        df (pd.DataFrame): DataFrame that includes 'fov' and measurement columns.
        layout (np.ndarray): Output from generate_chip_layout().
        parameter (str): Name of the column to map onto the chip.

    Returns:
        np.ndarray: 2D array of parameter values arranged by FOV layout.
        
    Raises:
        ValueError: If inputs are invalid or parameter not found in DataFrame.
        TypeError: If inputs are of wrong type.
    """
    try:
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
        
        if not isinstance(layout, np.ndarray):
            raise TypeError(f"layout must be a numpy array, got {type(layout).__name__}")
        
        if not isinstance(parameter, str):
            raise TypeError(f"parameter must be a string, got {type(parameter).__name__}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if parameter not in df.columns:
            raise ValueError(f"Parameter '{parameter}' not found in DataFrame columns. Available columns: {list(df.columns)}")
        
        if 'fov' not in df.columns:
            raise ValueError("DataFrame must contain 'fov' column")
        
        if layout.size == 0:
            raise ValueError("Layout array is empty")

        rows, cols = layout.shape
        chip_values = np.full((rows, cols), np.nan)

        try:
            # Calculate the mean value for each FOV
            df_unique = df.groupby('fov', as_index=False)[parameter].mean()
            
            if df_unique.empty:
                print(f"⚠️ Warning: No data found after grouping by FOV for parameter '{parameter}'")
                return chip_values
                
        except Exception as e:
            print(f"⚠️ Warning: Error calculating FOV means for parameter '{parameter}': {str(e)}")
            return chip_values

        fov_to_value = dict(zip(df_unique['fov'], df_unique[parameter]))

        # Fill chip_values
        valid_fovs = 0
        for row in range(rows):
            for col in range(cols):
                try:
                    fov = layout[row, col]
                    if fov >= 0 and fov in fov_to_value:  # Check for valid FOV (>= 0)
                        chip_values[row, col] = fov_to_value[fov]
                        valid_fovs += 1
                except (IndexError, KeyError) as e:
                    print(f"⚠️ Warning: Error accessing layout[{row}, {col}]: {str(e)}")
                    continue

        print(f"✅ Successfully mapped {valid_fovs} FOVs for parameter '{parameter}'")
        return chip_values
        
    except Exception as e:
        print(f"❌ Error mapping FOV to chip: {str(e)}")
        raise


def plot_chip_heatmap_bokeh(sample, df, parameter, scale='linear', title="Chip Heatmap"):
    """
    Create a Bokeh heatmap visualization of chip layout data.
    
    Parameters:
        sample: Sample object containing metadata
        df (pd.DataFrame): DataFrame with FOV and parameter data
        parameter (str): Column name to visualize
        scale (str): Scale type ('linear' or 'log')
        title (str): Plot title
        
    Returns:
        None: Displays the plot
        
    Raises:
        ValueError: If required data is missing or invalid
        Exception: For other plotting errors
    """
    try:
        # Input validation
        if sample is None:
            raise ValueError("Sample object cannot be None")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if not isinstance(parameter, str):
            raise TypeError(f"parameter must be a string, got {type(parameter).__name__}")
        
        if parameter not in df.columns:
            raise ValueError(f"Parameter '{parameter}' not found in DataFrame columns")

        try:
            output_notebook()
        except Exception as e:
            print(f"⚠️ Warning: Error setting up notebook output: {str(e)}")

        df = df.copy()
        
        # Get magnification from sample metadata
        try:
            if not hasattr(sample, 'sample_metadata') or sample.sample_metadata is None:
                raise ValueError("Sample object missing sample_metadata")
            
            if 'magnification' not in sample.sample_metadata:
                raise ValueError("Magnification not found in sample metadata")
                
            Objective = sample.sample_metadata['magnification']
            print(f"📊 Using objective magnification: {Objective}x")
            
        except Exception as e:
            print(f"⚠️ Warning: Error getting magnification from sample: {str(e)}")
            print("🔄 Defaulting to 4x magnification")
            Objective = 4

        # Filter CTCF values if needed
        try:
            if "ctcf" in parameter.lower():
                df = df.copy()
                original_count = len(df)
                df[parameter] = pd.to_numeric(df[parameter], errors="coerce")
                df.loc[df[parameter] < 0, parameter] = np.nan
                valid_count = df[parameter].notna().sum()
                print(f"📈 CTCF filtering: {valid_count}/{original_count} values are valid")
        except Exception as e:
            print(f"⚠️ Warning: Error filtering CTCF values: {str(e)}")

        # Generate layout and chip matrix
        try:
            layout = generate_chip_layout(Objective)
            chip_values = map_fov_to_chip(df, layout, parameter)
        except Exception as e:
            print(f"❌ Error generating chip layout or mapping values: {str(e)}")
            return

        rows, cols = chip_values.shape
        chip_values_flipped = np.flipud(chip_values)
        
        try:
            fov_counts = df['fov'].value_counts().to_dict()
        except Exception as e:
            print(f"⚠️ Warning: Error calculating FOV counts: {str(e)}")
            fov_counts = {}

        # Prepare data for Bokeh
        data = {
            "x": [],
            "y": [],
            "fov": [],
            "value": [],
            "N": []
        }

        try:
            for row in range(rows):
                for col in range(cols):
                    val = chip_values_flipped[row, col]
                    data["x"].append(col)
                    data["y"].append(row)
                    data["value"].append(val)
                    # Get FOV from original layout (also flip y)
                    fov = layout[rows - 1 - row, col]
                    data["fov"].append(fov)
                    data["N"].append(fov_counts.get(fov, 0))
        except Exception as e:
            print(f"❌ Error preparing plot data: {str(e)}")
            return

        try:
            source = ColumnDataSource(data)
        except Exception as e:
            print(f"❌ Error creating Bokeh data source: {str(e)}")
            return

        # Check if all values are NaN (avoid crash)
        valid_values = [v for v in data["value"] if not np.isnan(v)]
        if not valid_values:
            print(f"⚠️ All values are NaN for parameter '{parameter}'. Nothing to plot.")
            return

        # Compute vmin, vmax
        try:
            vmin = np.nanmin(data["value"])
            vmax = np.nanmax(data["value"])
            
            if np.isnan(vmin) or np.isnan(vmax):
                print(f"⚠️ Warning: Invalid min/max values: min={vmin}, max={vmax}")
                return
                
            print(f"📊 Value range: {vmin:.3f} to {vmax:.3f}")
            
        except Exception as e:
            print(f"❌ Error calculating value range: {str(e)}")
            return

        # Choose color mapper
        try:
            if "ctcf" in parameter.lower() and vmin > 0:
                color_mapper = LogColorMapper(palette=Viridis256, low=max(vmin, 0.001), high=vmax)
                print("🎨 Using logarithmic color scale for CTCF data")
            else:
                color_mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)
                print("🎨 Using linear color scale")
        except Exception as e:
            print(f"⚠️ Warning: Error creating color mapper, using default: {str(e)}")
            color_mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)

        # Create the plot
        try:
            TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

            p = figure(
                title=title,
                x_range=Range1d(-1, cols + 1), y_range=Range1d(-1, rows + 1),
                tools=TOOLS,
                match_aspect=True,
                tooltips=[("FOV", "@fov"), ("Mean Value", "@value{0.00}"), ("N", "@N")],
                width=750, height=800,
                toolbar_location='below'
            )

            p.rect(x="x", y="y", width=1, height=1, source=source,
                   fill_color=transform("value", color_mapper),
                   line_color=None)

            p.add_layout(ColorBar(color_mapper=color_mapper, location=(0, 0)), 'right')
            p.grid.grid_line_color = None
            p.axis.visible = False
            p.axis.major_label_text_font_size = "10pt"
            p.title.text_font_size = "16pt"
            p.title.align = "center"
            p.title.text_font_style = "bold"

            handle = show(p, notebook_handle=True)
            push_notebook(handle=handle)
            
        except Exception as e:
            print(f"❌ Error creating or displaying plot: {str(e)}")
            return
            
    except Exception as e:
        print(f"❌ Critical error in plot_chip_heatmap_bokeh: {str(e)}")
        import traceback
        traceback.print_exc()
        return


def chip_heatmap_controls(filtered_df, sample, data_type="wells_data"):
    """
    Create interactive controls for heatmap parameters.
    
    Parameters:
        filtered_df (pd.DataFrame): DataFrame to extract options from
        sample: Sample object containing channel information
        data_type (str): The data type key to use for PARQUET_FILES lookup
        
    Returns:
        None: Displays the control widgets
        
    Raises:
        ValueError: If DataFrame is invalid or missing required columns
    """
    try:
        global selected_title, selected_channel, selected_timepoint, selected_feature   

        # Input validation
        if not isinstance(filtered_df, pd.DataFrame):
            raise TypeError(f"filtered_df must be a pandas DataFrame, got {type(filtered_df).__name__}")
        
        if filtered_df.empty:
            raise ValueError("DataFrame is empty")
        
        required_columns = ['sequence']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        try:
            timepoints = sorted(filtered_df['sequence'].unique().astype(int))
            if not timepoints:
                raise ValueError("No timepoints found in sequence column")
        except Exception as e:
            print(f"⚠️ Warning: Error extracting timepoints: {str(e)}")
            timepoints = [0]  # Default fallback

        # Create widgets with error handling
        try:
            title_widget = widgets.Text(
                placeholder='Enter plot title...',
                description='Title:',
                layout=widgets.Layout(width='400px')
            )

            # Get available channels from data and sample
            available_channel_indices = sorted(filtered_df['channel_index'].unique().tolist())
            
            # Create channel options showing both channel number and name from sample.channels
            channel_options = []
            for channel_idx in available_channel_indices:
                if sample and hasattr(sample, 'channels') and channel_idx in sample.channels:
                    channel_name = sample.channels[channel_idx]
                    channel_options.append((f"{channel_idx}: {channel_name}", channel_idx))
                else:
                    channel_options.append((f"Channel {channel_idx}", channel_idx))
            
            channel_widget = widgets.Dropdown(
                options=channel_options,
                description='Channel:'
            )

            # Validate feature columns exist
            available_features = [f for f in PARQUET_FILES[data_type]["feature_columns"] if f in filtered_df.columns]
            if not available_features:
                print(f"⚠️ Warning: No feature columns found. Available columns: {list(filtered_df.columns)}")
                available_features = ['placeholder_feature']

            feature_widget = widgets.Dropdown(
                options=available_features,
                description='Feature:'
            )

            timepoint_widget = Dropdown(
                options=timepoints,
                value=timepoints[0] if timepoints else None,
                description='Sequence:'
            )

            # Create button for generating heatmap
            generate_button = widgets.Button(
                description='🗺️ Generate Heatmap',
                button_style='primary',
                layout=widgets.Layout(width='200px', height='40px')
            )
            
            # Create output area for heatmap display
            output_area = widgets.Output()

            # timepoint_widget = widgets.IntSlider(
                # min=min(timepoints),
                # max=max(timepoints),
                # step=1,
                # value=timepoints[0],
                # description='Sequence:',
                # continuous_update=True
            # )
        except Exception as e:
            print(f"❌ Error creating widgets: {str(e)}")
            return

        # Save parameters on interaction with error handling
        def save_parameters(change=None):
            try:
                global selected_title, selected_channel, selected_timepoint, selected_feature
                selected_title = title_widget.value
                selected_channel = channel_widget.value
                selected_timepoint = timepoint_widget.value
                selected_feature = feature_widget.value
                
            except Exception as e:
                print(f"⚠️ Warning: Error saving parameters: {str(e)}")

        # Button click handler
        def on_generate_click(button):
            """Generate heatmap when button is clicked."""
            # Clear the output area first
            output_area.clear_output()
            
            # Save current parameters
            save_parameters()
            
            # Show status in output area
            with output_area:
                print("🗺️ Generating heatmap...")
            
            # Get channel name for title
            channel_name_for_title = selected_channel  # Default to channel index
            if sample and hasattr(sample, 'channels') and selected_channel in sample.channels:
                channel_name_for_title = sample.channels[selected_channel]
            
            # Call run_heatmap outside the output context so Bokeh plot displays properly
            try:
                run_heatmap(filtered_df, sample, selected_feature, selected_title, selected_timepoint, selected_channel, channel_name_for_title)
                
                # Show success message in output area
                with output_area:
                    output_area.clear_output()
                    print("✅ Heatmap generated successfully!")
            except Exception as e:
                # Show error in output area
                with output_area:
                    output_area.clear_output()
                    print(f"❌ Error generating heatmap: {str(e)}")

        # Attach handlers with error handling
        try:
            title_widget.observe(save_parameters, names='value')
            channel_widget.observe(save_parameters, names='value')
            timepoint_widget.observe(save_parameters, names='value')
            feature_widget.observe(save_parameters, names='value')
            generate_button.on_click(on_generate_click)
        except Exception as e:
            print(f"⚠️ Warning: Error attaching event handlers: {str(e)}")

        # Show controls with error handling
        try:
            display(widgets.VBox([
                title_widget,
                widgets.HBox([channel_widget, feature_widget]),
                timepoint_widget,
                generate_button,
                output_area
            ]))
        except Exception as e:
            print(f"❌ Error displaying controls: {str(e)}")
            
    except Exception as e:
        print(f"❌ Critical error in chip_heatmap_controls: {str(e)}")
        import traceback
        traceback.print_exc()


def run_heatmap(filtered_df, sample, selected_feature, selected_title, selected_timepoint, selected_channel, channel_name_for_title=None):
    """
    Execute heatmap generation with comprehensive error handling.
    
    Parameters:
        filtered_df (pd.DataFrame): DataFrame with data
        sample: Sample object
        selected_feature (str): Feature to plot
        selected_title (str): Plot title
        selected_timepoint (int): Timepoint to filter
        selected_channel (int): Channel to filter
        channel_name_for_title (str, optional): Channel name for display in title
        
    Returns:
        None: Displays the heatmap or error message
    """
    try:
        # Check if filtered_df exists and is valid
        if 'filtered_df' not in globals() or filtered_df is None:
            print("❌ Please run the classification filter cell first.")
            return
        
        if not isinstance(filtered_df, pd.DataFrame):
            print(f"❌ filtered_df must be a DataFrame, got {type(filtered_df).__name__}")
            return
            
        if filtered_df.empty:
            print("❌ DataFrame is empty.")
            return
        
        # Validate required columns
        required_columns = ['channel_index', 'sequence']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"❌ DataFrame missing required columns: {missing_columns}")
            return
        
        # Filter data based on selection
        try:
            data_subset = filtered_df[
                (filtered_df['channel_index'] == selected_channel) & 
                (filtered_df['sequence'] == selected_timepoint)
            ]
            
            if data_subset.empty:
                print(f"❌ No data available for the selected combination:")
                print(f"   Channel: {selected_channel}, Timepoint: {selected_timepoint}")
                print(f"   Available channels: {sorted(filtered_df['channel_index'].unique())}")
                print(f"   Available sequences: {sorted(filtered_df['sequence'].unique())}")
                return
                
        except Exception as e:
            print(f"❌ Error filtering data: {str(e)}")
            return
        
        # Validate feature exists in data
        if selected_feature not in data_subset.columns:
            print(f"❌ Feature '{selected_feature}' not found in data.")
            print(f"   Available features: {[col for col in data_subset.columns if col in PARQUET_FILES['wells_data']['feature_columns']]}")
            return
        
        # Create plot title using channel name instead of channel number
        try:
            # Use channel name for title if provided, otherwise fall back to channel index
            channel_display = channel_name_for_title if channel_name_for_title else f"Channel {selected_channel}"
            plot_title = f"{selected_title}, Sequence {selected_timepoint}, {channel_display}, {selected_feature.replace('_',' ')}"
        except Exception as e:
            print(f"⚠️ Warning: Error creating plot title: {str(e)}")
            plot_title = f"Heatmap - {selected_feature}"
        
        print(f"🎯 Creating heatmap for:")
        print(f"   Feature: {selected_feature}")
        print(f"   Sequence: {selected_timepoint}")
        print(f"   Channel: {selected_channel}")
        print(f"   Data points: {len(data_subset)}")
        
        # Generate the heatmap
        plot_chip_heatmap_bokeh(
            sample=sample,
            df=data_subset,
            parameter=selected_feature,
            title=plot_title
        )
        
    except Exception as e:
        print(f"❌ Critical error in run_heatmap: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_interactive_scatter(df, controls, eps=300, min_samples=5):
    """
    Create an interactive Bokeh scatter plot with selection and saving capabilities.
    
    Parameters:
        df (pd.DataFrame): DataFrame with data
        controls (dict): Dictionary of control widgets
        eps (float): DBSCAN epsilon parameter
        min_samples (int): DBSCAN minimum samples parameter
        
    Returns:
        dict: Dictionary of saved queries
        
    Raises:
        ValueError: If inputs are invalid or missing required data
    """
    try:
        global selected_global_indices
        selected_global_indices = []

        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if not isinstance(controls, dict):
            raise TypeError(f"controls must be a dictionary, got {type(controls).__name__}")
    

        # Check required control keys based on comparison mode
        basic_keys = ['classification_mode', 'sequence', 'run_clustering', 'fov', 'title']
        missing_keys = [key for key in basic_keys if key not in controls]
        if missing_keys:
            raise ValueError(f"Missing required control keys: {missing_keys}")

        # Extract basic values from controls with error handling
        try:
            comparison_mode = controls['classification_mode']
            sequence = controls['sequence'].value
            run_clustering = controls['run_clustering'].value
            selected_fovs = list(controls['fov'].value)

            if sequence is None:
                raise ValueError("Sequence cannot be None")
                
        except Exception as e:
            print(f"❌ Error extracting basic control values: {str(e)}")
            return {}

        # Extract mode-specific values
        try:
            if comparison_mode == 'features':
                # Feature comparison mode: X and Y are different features, single channel
                if 'channel' not in controls or 'x_feature' not in controls or 'y_feature' not in controls:
                    raise ValueError("Feature comparison mode requires 'channel', 'x_feature', and 'y_feature' controls")
                
                channel = controls['channel'].value
                x_feature = controls['x_feature'].value
                y_feature = controls['y_feature'].value
                x_channel = y_channel = channel  # Same channel for both axes
                single_feature = None
                
                if not all([x_feature, y_feature, channel is not None]):
                    raise ValueError("Feature comparison mode: x_feature, y_feature, and channel cannot be None")
                
                print(f"🎯 Feature Comparison Mode:")
                print(f"   Channel: {channel}")
                print(f"   X Feature: {x_feature}")
                print(f"   Y Feature: {y_feature}")
                
            else:  # channels
                # Channel comparison mode: X and Y are different channels, single feature
                if 'feature' not in controls or 'x_channel' not in controls or 'y_channel' not in controls:
                    raise ValueError("Channel comparison mode requires 'feature', 'x_channel', and 'y_channel' controls")
                
                single_feature = controls['feature'].value
                x_channel = controls['x_channel'].value
                y_channel = controls['y_channel'].value
                x_feature = y_feature = single_feature  # Same feature for both axes
                channel = None  # Not used in this mode
                
                if not all([single_feature, x_channel is not None, y_channel is not None]):
                    raise ValueError("Channel comparison mode: feature, x_channel, and y_channel cannot be None")
                
                print(f"🎯 Channel Comparison Mode:")
                print(f"   Feature: {single_feature}")
                print(f"   X Channel: {x_channel}")
                print(f"   Y Channel: {y_channel}")
                
        except Exception as e:
            print(f"❌ Error extracting mode-specific control values: {str(e)}")
            return {}

        # Validate required columns exist in DataFrame
        required_columns = ['sequence', 'fov', 'global_index', x_feature, y_feature]
        
        # Add channel_index column requirement for channel comparison mode
        if comparison_mode == 'channels':
            required_columns.append('channel_index')
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ DataFrame missing required columns: {missing_columns}")
            print(f"   Available columns: {list(df.columns)}")
            return {}

        print(f"   Sequence: {sequence}")
        print(f"   Selected FOVs: {selected_fovs}")

        # Filter data with error handling
        try:
            df_filtered = df[df['sequence'] == sequence].copy()
            
            if df_filtered.empty:
                print(f"❌ No data found for sequence {sequence}")
                print(f"   Available sequences: {sorted(df['sequence'].unique())}")
                return {}
                
            if 'All' not in selected_fovs:
                selected_fovs = [f for f in selected_fovs if f != 'All']
                if not selected_fovs:
                    print("❌ No FOVs selected.")
                    return {}
                    
                original_count = len(df_filtered)
                df_filtered = df_filtered[df_filtered['fov'].isin(selected_fovs)].copy()
                
                if df_filtered.empty:
                    print(f"❌ No data found for selected FOVs: {selected_fovs}")
                    print(f"   Available FOVs: {sorted(df['fov'].unique())}")
                    return {}
                    
                print(f"📊 FOV filtering: {len(df_filtered)}/{original_count} rows retained")

            # Apply mode-specific filtering
            if comparison_mode == 'features':
                # Feature comparison: filter by single channel
                if 'channel_index' in df_filtered.columns:
                    original_count = len(df_filtered)
                    df_filtered = df_filtered[df_filtered['channel_index'] == channel].copy()
                    print(f"📊 Channel filtering: {len(df_filtered)}/{original_count} rows retained for channel {channel}")
                else:
                    print("⚠️ Warning: 'channel_index' column not found, skipping channel filtering")
                    
            else:  # channel comparison
                # Channel comparison: we need data from both channels, will process separately
                print(f"📊 Channel comparison: will process channels {x_channel} and {y_channel} separately")
                
        except Exception as e:
            print(f"❌ Error filtering data by sequence/FOV: {str(e)}")
            return {}

        # Prepare data based on comparison mode
        try:
            if comparison_mode == 'features':
                # Feature comparison mode: use filtered data as-is
                final_df = df_filtered.copy()
                
                # Feature cleanup with error handling
                for feature in [x_feature, y_feature]:
                    if "ctcf" in feature.lower():
                        original_count = len(final_df)
                        final_df[feature] = pd.to_numeric(final_df[feature], errors="coerce")
                        final_df.loc[final_df[feature] < 0, feature] = np.nan
                        valid_count = final_df[feature].notna().sum()
                        print(f"🧹 CTCF cleanup for {feature}: {valid_count}/{original_count} values are valid")
                
                # Drop rows with missing values in required columns
                columns_to_check = [x_feature, y_feature, 'global_index', 'fov']
                original_count = len(final_df)
                final_df = final_df[columns_to_check].dropna()
                
                if final_df.empty:
                    print(f"❌ No data remaining after removing missing values.")
                    print(f"   Original count: {original_count}")
                    return {}
                    
                print(f"🔍 Data cleaning: {len(final_df)}/{original_count} rows have complete data")
                
            else:  # channel comparison mode
                # Channel comparison: merge data from two channels
                if 'channel_index' not in df_filtered.columns:
                    print("❌ Error: 'channel_index' column required for channel comparison")
                    return {}
                
                # Get data for X channel
                df_x = df_filtered[df_filtered['channel_index'] == x_channel].copy()
                if df_x.empty:
                    print(f"❌ No data found for X channel {x_channel}")
                    return {}
                
                # Get data for Y channel  
                df_y = df_filtered[df_filtered['channel_index'] == y_channel].copy()
                if df_y.empty:
                    print(f"❌ No data found for Y channel {y_channel}")
                    return {}
                
                # Feature cleanup for the single feature
                if "ctcf" in single_feature.lower():
                    for channel_df, channel_name in [(df_x, f"X channel {x_channel}"), (df_y, f"Y channel {y_channel}")]:
                        original_count = len(channel_df)
                        channel_df[single_feature] = pd.to_numeric(channel_df[single_feature], errors="coerce")
                        channel_df.loc[channel_df[single_feature] < 0, single_feature] = np.nan
                        valid_count = channel_df[single_feature].notna().sum()
                        print(f"🧹 CTCF cleanup for {channel_name}: {valid_count}/{original_count} values are valid")
                
                # Clean data: remove NaN values for the feature of interest
                df_x_clean = df_x.dropna(subset=[single_feature]).copy()
                df_y_clean = df_y.dropna(subset=[single_feature]).copy()
                
                # Merge on common columns to get paired data points
                merge_cols = ['global_index', 'sequence', 'fov']
                merge_cols = [col for col in merge_cols if col in df_x_clean.columns and col in df_y_clean.columns]
                
                if not merge_cols:
                    print("❌ No common columns found for merging channel data")
                    return {}
                
                # Merge the two channel datasets - pandas will add _x and _y suffixes to distinguish columns
                final_df = pd.merge(df_x_clean, df_y_clean, on=merge_cols, how='inner')
                
                if final_df.empty:
                    print(f"❌ No matching data points between channels {x_channel} and {y_channel}")
                    return {}
                
                print(f"🔍 Channel comparison: {len(final_df)} data points with both channels")
                print(f"   X channel {x_channel}: {len(df_x_clean)} points")
                print(f"   Y channel {y_channel}: {len(df_y_clean)} points")
                print(f"   Overlapping: {len(final_df)} points")
                
        except Exception as e:
            print(f"❌ Error preparing data: {str(e)}")
            return {}

        # Clustering with error handling
        try:
            if run_clustering:
                if comparison_mode == 'features':
                    clustering_data = final_df[[x_feature, y_feature]].dropna()
                else:  # channels - use the suffixed column names after merge
                    clustering_data = final_df[[f"{single_feature}_x", f"{single_feature}_y"]].dropna()
                
                if len(clustering_data) >= min_samples:
                    clustering = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = clustering.fit_predict(clustering_data)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    final_df['cluster'] = cluster_labels.astype(str)
                    final_df.loc[final_df['cluster'] == '-1', 'cluster'] = 'noise'
                    print(f"🎯 Clustering results: {n_clusters} clusters, {n_noise} noise points")
                else:
                    print(f"⚠️ Warning: Not enough data points ({len(clustering_data)}) for clustering (min_samples={min_samples})")
                    print("🔄 Disabling clustering")
                    final_df['cluster'] = "0"
            else:
                final_df['cluster'] = "0"
                print("🎯 Clustering disabled - all points in single group")
                
        except Exception as e:
            print(f"⚠️ Warning: Error in clustering, using single cluster: {str(e)}")
            final_df['cluster'] = "0"

        # Prepare plotting variables
        try:
            unique_clusters = sorted(final_df['cluster'].unique())
            palette = Category10[10] * ((len(unique_clusters) // 10) + 1)
            
            # Create data source
            source = ColumnDataSource(final_df)
            
            # Determine scales
            x_scale = LogScale() if "ctcf" in x_feature.lower() else LinearScale()
            y_scale = LogScale() if "ctcf" in y_feature.lower() else LinearScale()
            
            # Create title based on comparison mode
            fov_label = "All" if 'All' in controls['fov'].value else ','.join(map(str, selected_fovs[:3])) + ("..." if len(selected_fovs) > 3 else "")
            
            if comparison_mode == 'features':
                title = f"{controls['title'].value} Seq {sequence}, Channel {channel}, FOV {fov_label}"
            else:  # channels
                title = f"{controls['title'].value} Seq {sequence}, FOV {fov_label} - ({single_feature})"
            
        except Exception as e:
            print(f"❌ Error preparing plot variables: {str(e)}")
            return {}

        # Create the plot
        try:
            # Create appropriate axis labels and tooltips
            if comparison_mode == 'features':
                x_label = x_feature.replace("_", " ")
                y_label = y_feature.replace("_", " ")
                # In feature comparison mode, use original feature names
                x_feature_display = x_feature  
                y_feature_display = y_feature
                tooltips = [
                    ("FOV", "@fov"),
                    ("Global Index", "@global_index"),
                    (x_feature, f"@{x_feature}"),
                    (y_feature, f"@{y_feature}"),
                    ("Cluster", "@cluster")
                ]
            else:  # channels
                x_label = f"{single_feature.replace('_', ' ')} (Channel {x_channel})"
                y_label = f"{single_feature.replace('_', ' ')} (Channel {y_channel})"
                
                # In channel comparison mode, the merge adds _x and _y suffixes to feature names
                
                x_feature_display = f"{single_feature}_x"  # Column name after merge
                y_feature_display = f"{single_feature}_y"  # Column name after merge
                tooltips = [
                    ("FOV", "@fov"),
                    ("Global Index", "@global_index"),
                    (f"Ch {x_channel}", f"@{x_feature_display}"),
                    (f"Ch {y_channel}", f"@{y_feature_display}"),
                    ("Cluster", "@cluster")
                ]

            p = figure(
                title=title,
                height=700,
                width=700,
                tools=["pan", "wheel_zoom", "reset", "box_select", "lasso_select", "tap", "hover", "save"],
                tooltips=tooltips,
                x_scale=x_scale,
                y_scale=y_scale
            )

            p.scatter(
                x=x_feature_display if comparison_mode == 'channels' else x_feature,
                y=y_feature_display if comparison_mode == 'channels' else y_feature,
                source=source,
                size=6,
                alpha=0.7,
                color=factor_cmap('cluster', palette=palette, factors=unique_clusters)
            )

            p.xaxis.axis_label = x_label
            p.yaxis.axis_label = y_label
            p.title.text_font_size = "16pt"
            p.title.align = "center"
            p.title.text_font_style = "bold"
            
        except Exception as e:
            print(f"❌ Error creating Bokeh plot: {str(e)}")
            return {}

        # Create selection interface
        try:
            # JS textbox for copy
            bokeh_textbox = TextAreaInput(value="", rows=4, title="Selected Indices:")
            source.selected.js_on_change("indices", CustomJS(args=dict(source=source, textbox=bokeh_textbox), code="""
                const inds = cb_obj.indices;
                const data = source.data;
                const selected = inds.map(i => data['global_index'][i]);
                textbox.value = selected.join(",");
            """))

            # Save UI
            group_name_input = widgets.Text(
                placeholder="Enter query name", 
                description="Query Name:", 
                layout=widgets.Layout(width='300px')
            )
            group_description_input = widgets.Textarea(
                placeholder="Enter a short description", 
                description="Description:", 
                layout=widgets.Layout(width='500px')
            )
            python_textbox = widgets.Textarea(
                placeholder="Paste global indices here", 
                description="Indices:", 
                layout=widgets.Layout(width='500px')
            )
            save_button = widgets.Button(description="Save Query", button_style='success')
            output_area = widgets.Output()

            saved_queries = {}

            def on_save_click(b):
                nonlocal saved_queries
                global selected_global_indices

                try:
                    name = group_name_input.value.strip()
                    desc = group_description_input.value.strip()
                    text = python_textbox.value.strip()

                    if not name:
                        with output_area:
                            clear_output()
                            print("❌ Error: Query name is required")
                        return
                    
                    if not text:
                        with output_area:
                            clear_output()
                            print("❌ Error: No indices provided")
                        return

                    selected_global_indices = [int(i.strip()) for i in text.split(",") if i.strip().isdigit()]
                    
                    if not selected_global_indices:
                        with output_area:
                            clear_output()
                            print("❌ Error: No valid indices found")
                        return
                    
                    saved_queries[name] = {
                        'indices': selected_global_indices,
                        'description': desc
                    }
                    
                    with output_area:
                        clear_output()
                        print(f"✅ Saved '{name}' with {len(selected_global_indices)} indices.")
                        
                    # Clear the form
                    group_name_input.value = ""
                    group_description_input.value = ""
                    python_textbox.value = ""
                    
                except ValueError as e:
                    with output_area:
                        clear_output()
                        print(f"❌ Error parsing indices: {str(e)}")
                except Exception as e:
                    with output_area:
                        clear_output()
                        print(f"❌ Error saving query: {str(e)}")

            save_button.on_click(on_save_click)

            # Show Bokeh + save UI
            show(column(p, bokeh_textbox))
            display(widgets.VBox([
                group_name_input,
                group_description_input,
                python_textbox,
                save_button,
                output_area
            ]))

            return saved_queries
            
        except Exception as e:
            print(f"❌ Error creating selection interface: {str(e)}")
            return {}
            
    except Exception as e:
        print(f"❌ Critical error in plot_interactive_scatter: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def create_scatter_controls(df, mode, sample, data_type="wells_data"):
    """Create enhanced controls for the selected comparison mode with channel names and SaveTool"""
    
    try:
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        # Get available features and channels
        feature_cols = [col for col in df.columns if col in PARQUET_FILES[data_type]["feature_columns"]]
        available_channels = sorted(df['channel_index'].unique()) if 'channel_index' in df.columns else []
        
        if not feature_cols:
            raise ValueError("No feature columns found in DataFrame")
        if not available_channels:
            raise ValueError("No channel data found in DataFrame")
        
        # Get channel mapping from sample
        channel_mapping = sample.channels
        
        # Create channel options with both number and value (e.g., "2: CY5")
        channel_options = []
        for idx in available_channels:
            channel_name = channel_mapping.get(idx, f"Channel {idx}")
            channel_options.append((f"{idx}: {channel_name}", idx))
        
        # Create original controls - FOV, Sequence, and DBSCAN clustering
        sequences = sorted(df['sequence'].unique()) if 'sequence' in df.columns else [1]
        fov_options = ['All'] + sorted(df['fov'].unique().tolist()) if 'fov' in df.columns else ['All']
        
        title_widget = Text(placeholder='Enter plot title...', description='Title:', layout={'width': '400px'})

        sequence_widget = widgets.Dropdown(
            options=sequences,
            value=sequences[0] if sequences else None,
            description='Sequence:',
            layout={'width': '200px'}
        )
        
        fov_widget = widgets.SelectMultiple(
            options=fov_options,
            value=['All'],
            description='FOV(s):',
            layout={'width': '250px', 'height': '120px'}
        )

        run_clustering_widget = widgets.ToggleButtons(
            options=[("Yes", True), ("No", False)],
            value=True,
            description="Auto-Clustering:",
            style={'description_width': 'initial'}
        )
        
        controls = {
            'classification_mode': mode,
            'title': title_widget,
            'sequence': sequence_widget,
            'fov': fov_widget,
            'run_clustering': run_clustering_widget
        }
        
        if mode == 'features':
            # Feature comparison mode: X and Y are different features, single channel
            x_feature_widget = widgets.Dropdown(
                options=feature_cols,
                value=feature_cols[0],
                description='X Feature:',
                style={'description_width': 'initial'}
            )
            
            y_feature_widget = widgets.Dropdown(
                options=feature_cols,
                value=feature_cols[1] if len(feature_cols) > 1 else feature_cols[0],
                description='Y Feature:',
                style={'description_width': 'initial'}
            )
            
            # Enhanced channel widget with both number and value
            channel_widget = widgets.Dropdown(
                options=channel_options,
                value=available_channels[0],
                description='Channel:',
                style={'description_width': 'initial'}
            )
            
            controls.update({
                'x_feature': x_feature_widget,
                'y_feature': y_feature_widget,
                'channel': channel_widget  # This is what the plotting function expects
            })
            
            display(widgets.VBox([
                widgets.HTML(f"<b>Feature Comparison Settings:</b>"),
                x_feature_widget,
                y_feature_widget,
                channel_widget,
                widgets.HTML("<b>General Settings:</b>"),
                title_widget,
                sequence_widget,
                fov_widget,
                run_clustering_widget
            ]))
            
        else:  # channels mode
            # Channel comparison mode: select one feature, plus X and Y channels
            feature_widget = widgets.Dropdown(
                options=feature_cols,
                value=feature_cols[0],
                description='Feature:',
                style={'description_width': 'initial'}
            )
            
            # Enhanced channel widgets with both number and value
            x_channel_widget = widgets.Dropdown(
                options=channel_options,
                value=available_channels[0],
                description='X Channel:',
                style={'description_width': 'initial'}
            )
            
            y_channel_widget = widgets.Dropdown(
                options=channel_options,
                value=available_channels[1] if len(available_channels) > 1 else available_channels[0],
                description='Y Channel:',
                style={'description_width': 'initial'}
            )
            
            controls.update({
                'feature': feature_widget,
                'x_channel': x_channel_widget,
                'y_channel': y_channel_widget,
                'channel': x_channel_widget  # Default to x_channel for backward compatibility
            })
            
            display(widgets.VBox([
                widgets.HTML(f"<b>Channel Comparison Settings:</b>"),
                feature_widget,
                x_channel_widget,
                y_channel_widget,
                widgets.HTML("<b>General Settings:</b>"),
                title_widget,
                sequence_widget,
                fov_widget,
                run_clustering_widget
            ]))
        
        return controls
        
    except Exception as e:
        print(f"❌ Error creating scatter controls: {e}")
        return None


def create_comparison_mode_selector():
    """Create and display scatter plot comparison mode selector widget."""
    global comparison_mode_widget
    comparison_mode_widget = widgets.ToggleButtons(
        options=[("Feature Comparison", "features"), ("Channel Comparison", "channels")],
        value="features",
        description="Comparison Mode:",
        style={'description_width': 'initial'}
    )
    display(widgets.VBox([
        widgets.HTML("<h3>🎯 Scatter Plot Configuration</h3>"),
        widgets.HTML("<p>Choose how you want to compare your data:</p>"),
        widgets.HTML("<b>• Feature Comparison:</b> Compare different features (e.g., intensity vs area) for one channel"),
        widgets.HTML("<b>• Channel Comparison:</b> Compare same feature between different channels"),
        comparison_mode_widget
    ]))
    return comparison_mode_widget


def plot_multi_timepoint_hist(
    df,  # Your data table (must have measurement and time columns)
    channel_mapping, # the channels mapping (e.g., {0: "Brightfield", 1: "GFP", 2: "mCherry"})
    feature_name,  # Name of the column with the measurement to plot (e.g., "Cell Area")
    selected_time_points = [0,1,2],  # List of time point values to include (e.g., [1, 3])
    channel=0,  # Channel number to filter by (e.g.,0 for Brightfield, 1 for GFP, 2 for mCherry)
    treatment='Name',  # Label for the treatment group (e.g., "EV1", "EV2", "Control")
    time_column='sequence',  # Name of the column indicating the time point (e.g., "Hours")
    x_scale='log',  # Scale for the x-axis: 'symlog', 'log', or 'linear'
    linthresh=1,  # Threshold for linear scaling of the x-axis (used for symlog)
    n_bins=50,  # Number of bars in the histogram (adjust for detail)
    font_size=11,  # Size of the text on the plot
    alpha=0.7,  # Transparency of the histogram bars
    cumulative=False,  # If True, shows a cumulative distribution instead of a histogram
    outline=False,  # If True, shows the outline of the histogram
    smooth=False,  # If True and outline=True, smooths the outline
    border_width=1.5,  # Thickness of the outline or CDF line
    constant_N=False,          # <<< NEW: force same N across selected time points
    random_seed=42,            # <<< NEW: reproducible downsampling
    hide_negative_ctcf=True    # <<< NEW: filter out negative CTCF values if True
):
    # --- Collect values per time point as Series indexed by global_index ---
    per_tp_series = {}   # time_point -> pd.Series(index=global_index, values=feature)
    Ns = {}              # time_point -> N before any downsampling
    all_data = []

    for time_point in selected_time_points:
        # rows for requested time & channel
        df_tp = df[(df[time_column] == time_point) & (df['channel_index'] == channel)]
        if df_tp.empty:
            print(f"Skipping time point {time_point} — channel {channel} not found.")
            per_tp_series[time_point] = None
            Ns[time_point] = 0
            continue

        # Keep only the necessary columns, drop NaN feature values, and de-duplicate per (gi, time, channel)
        cols_needed = ['global_index', feature_name, time_column, 'channel_index']
        cols_present = [c for c in cols_needed if c in df_tp.columns]
        d = df_tp[cols_present].dropna(subset=[feature_name]).copy()
        
        # Filter out negative CTCF values if requested
        if hide_negative_ctcf and "ctcf" in feature_name.lower():
            original_count = len(d)
            d = d[d[feature_name] > 0].copy()
            filtered_count = len(d)
            if original_count > filtered_count:
                print(f"Filtered out {original_count - filtered_count} negative CTCF values at time point {time_point}")
        
        d = d.sort_values(['global_index']).drop_duplicates(subset=['global_index'])  # one row per GI at this TP

        if d.empty:
            print(f"No data available for time point {time_point}.")
            per_tp_series[time_point] = None
            Ns[time_point] = 0
            continue

        s = d.set_index('global_index')[feature_name]  # Series with GI index
        per_tp_series[time_point] = s
        Ns[time_point] = len(s)
        all_data.extend(s.values.tolist())

    # If no data at all, exit early
    all_data = np.array(all_data)
    if all_data.size == 0:
        print("No data available for the selected feature and channel.")
        return

    # --- Optional: equalize N across time points by downsampling global_index ---
    if constant_N:
        
        positive_Ns = [n for n in Ns.values() if n > 0]
        if len(positive_Ns) == 0:
            print("No non-empty time points to equalize.")
            return
        target_N = int(np.min(positive_Ns))

        rng = np.random.default_rng(random_seed)

        # downsample each non-empty time point to target_N
        for tp, s in per_tp_series.items():
            if s is None or len(s) == 0:
                continue
            if len(s) > target_N:
                # sample global indices without replacement
                sampled_idx = rng.choice(s.index.values, size=target_N, replace=False)
                per_tp_series[tp] = s.loc[sampled_idx]
                Ns[tp] = target_N  # update N for labeling
            else:
                # already at or below target_N (should be equal for the min); if below (shouldn't),
                Ns[tp] = len(s)

        
        all_data = np.concatenate([s.values for s in per_tp_series.values() if s is not None and len(s) > 0])

    # --- Build bins ---
    data_min, data_max = all_data.min(), all_data.max()

    if len(selected_time_points) <= 10:
        bar_colors = Category10[max(3, len(selected_time_points))]
    else:
        # For more than 10 colors, cycle through Category10
        bar_colors = [Category10[10][i % 10] for i in range(len(selected_time_points))]

    if x_scale == 'symlog':
        bins = symlog_bins(all_data, linthresh=linthresh, n_bins=n_bins)
    elif x_scale == 'log':
        if np.any(all_data <= 0):
            print("Log scale requires all data to be positive.")
            return
        bins = log_bins(all_data, n_bins=n_bins)
    else:
        bins = np.linspace(data_min, data_max, n_bins + 1)

    
    # Set up the figure with only hover and zoom tools
    tools = ["hover", "box_zoom", "wheel_zoom", "pan", "reset","save"]
    
    # Determine scale
    if x_scale == 'log':
        x_scale_obj = LogScale()
    else:
        x_scale_obj = LinearScale()
    
    channel_name = channel_mapping.get(channel, f"Channel {channel}")
    title_suffix = " (equalized N)" if constant_N else ""
    plot_title = f"{treatment} - {channel_name} {feature_name.replace('_', ' ').title()} by Time Point{title_suffix}"
    
    p = figure(
        title=plot_title,
        x_axis_label=feature_name.replace('_',' '),
        y_axis_label="Fraction of Cells" if not cumulative else "Cumulative Fraction",
        width=1000,
        height=800,
        tools=tools,
        x_scale=x_scale_obj
    )

    # Configure hover tool to show time point information
    hover = p.select_one(HoverTool)
    hover.tooltips = [
        ("Time Point", "@timepoint"),
        ("Value", "@x"),
        ("Fraction", "@y{0.000}"),
        ("N", "@n_cells"),
        ("Mean", "@mean"),
        ("Median", "@median"),
        ("Std", "@std")
    ]

    # Plot each time point
    for i, time_point in enumerate(selected_time_points):
        s = per_tp_series.get(time_point, None)
        if s is None or len(s) == 0:
            print(f"No data available for time point {time_point}.")
            continue

        values = s.values
        N = len(values)
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)

        hist, bin_edges = np.histogram(values, bins=bins, weights=np.ones(len(values)) / len(values))

        if cumulative:
            cumulative_hist = np.cumsum(hist)
            cumulative_hist = cumulative_hist / cumulative_hist[-1]
            
            # Create line plot for cumulative distribution
            source = ColumnDataSource(data=dict(
                x=bin_edges[:-1],
                y=cumulative_hist,
                timepoint=[f"TP {time_point}"] * len(bin_edges[:-1]),
                n_cells=[N] * len(bin_edges[:-1]),
                mean=[f"{mean:.1f}"] * len(bin_edges[:-1]),
                median=[f"{median:.1f}"] * len(bin_edges[:-1]),
                std=[f"{std:.1f}"] * len(bin_edges[:-1])
            ))
            
            p.line('x', 'y', source=source, 
                  color=bar_colors[i % len(bar_colors)], 
                  line_width=border_width,
                  legend_label=f'TP {time_point} | N = {N} | CDF')
                  
        elif outline:
            midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            if smooth:
                # Create smooth line using scipy
                from scipy.interpolate import make_interp_spline
                x_smooth = np.linspace(midpoints.min(), midpoints.max(), 500)
                y_smooth = make_interp_spline(midpoints, hist, k=3)(x_smooth)
                
                source = ColumnDataSource(data=dict(
                    x=x_smooth,
                    y=y_smooth,
                    timepoint=[f"TP {time_point}"] * len(x_smooth),
                    n_cells=[N] * len(x_smooth),
                    mean=[f"{mean:.1f}"] * len(x_smooth),
                    median=[f"{median:.1f}"] * len(x_smooth),
                    std=[f"{std:.1f}"] * len(x_smooth)
                ))
            else:
                source = ColumnDataSource(data=dict(
                    x=midpoints,
                    y=hist,
                    timepoint=[f"TP {time_point}"] * len(midpoints),
                    n_cells=[N] * len(midpoints),
                    mean=[f"{mean:.1f}"] * len(midpoints),
                    median=[f"{median:.1f}"] * len(midpoints),
                    std=[f"{std:.1f}"] * len(midpoints)
                ))
            
            p.line('x', 'y', source=source,
                  color=bar_colors[i % len(bar_colors)], 
                  line_width=border_width,
                  legend_label=f'TP {time_point} | N = {N} | Mean={mean:.1f} | Median={median:.1f} | STD={std:.1f}')
        else:
            # Create histogram bars
            counts, bin_edges = np.histogram(values, bins=bins)
            fractions = counts / counts.sum()
            
            # Calculate bar properties
            left = bin_edges[:-1]
            right = bin_edges[1:]
            bottom = np.zeros(len(left))
            top = fractions
            
            source = ColumnDataSource(data=dict(
                left=left,
                right=right,
                bottom=bottom,
                top=top,
                x=(left + right) / 2,  # midpoint for hover
                y=top,
                timepoint=[f"TP {time_point}"] * len(left),
                n_cells=[N] * len(left),
                mean=[f"{mean:.1f}"] * len(left),
                median=[f"{median:.1f}"] * len(left),
                std=[f"{std:.1f}"] * len(left)
            ))
            
            p.quad(left='left', right='right', bottom='bottom', top='top',
                  source=source, color=bar_colors[i % len(bar_colors)], 
                  alpha=alpha, line_color="black", line_width=border_width,
                  legend_label=f'TP {time_point} | N = {N} | Mean={mean:.1f} | Median={median:.1f} | STD={std:.1f}')

    # Customize plot appearance
    p.legend.location = "top_right" 
    p.xgrid.grid_line_alpha = 0
    p.ygrid.grid_line_alpha = 0
    p.title.text_font_size = f"{font_size + 2}pt"
    p.xaxis.axis_label_text_font_size = f"{font_size}pt"
    p.yaxis.axis_label_text_font_size = f"{font_size}pt"
    p.legend.label_text_font_size = f"{font_size - 1}pt"
    p.legend.click_policy = "mute"  # Allow muting/unmuting series by clicking legend

    # Show the plot
    show(p)


def symlog_bins(data, linthresh=1, n_bins=100):
    bin_min, bin_max = data.min(), data.max()
    def symlog_transform(x):
        return np.sign(x) * np.log1p(np.abs(x) / linthresh)
    def symlog_inverse(y):
        return np.sign(y) * (np.expm1(np.abs(y)) * linthresh)
    transformed_min = symlog_transform(bin_min)
    transformed_max = symlog_transform(bin_max)
    transformed_bins = np.linspace(transformed_min, transformed_max, n_bins)
    return symlog_inverse(transformed_bins)


def log_bins(data, n_bins=50):
    """
    Creates log-spaced bins for strictly positive data (e.g., post-shifted CTCF features)

    Parameters:
        data (array-like): Input data, assumed to be all positive after preprocessing.
        n_bins (int): Number of bins.

    Returns:
        np.array: Log-spaced bin edges.
    """
    data = np.asarray(data)
    bin_min = data.min()
    bin_max = data.max()

    # Ensure strictly positive data for log binning
    if bin_min <= 0:
        raise ValueError("Data must be strictly positive for log binning.")

    # Create log-spaced bins
    log_bins = np.logspace(np.log10(bin_min), np.log10(bin_max), n_bins + 1)
    return log_bins


def create_multi_timepoint_hist_controls(df, sample, data_type="wells_data"):
    """Create interactive controls for multi-timepoint histogram plotting with enhanced button behavior."""
    try:
        # Get available features from the data
        available_features = [col for col in df.columns if col in PARQUET_FILES[data_type]["feature_columns"]]
        if not available_features:
            print("❌ No feature columns available in the data")
            return
        
        # Get available channels and time points
        available_channel_indices = sorted(df['channel_index'].unique()) if 'channel_index' in df.columns else []
        available_sequences = sorted(df['sequence'].unique()) if 'sequence' in df.columns else [0, 1, 2, 3, 4, 5]
        channel_mapping = sample.channels
        
        # Create channel options with names
        channel_options = []
        for idx in available_channel_indices:
            channel_name = channel_mapping.get(idx, f"Channel {idx}")
            channel_options.append((f"{idx}: {channel_name}", idx))
            
        # Create sequence options
        sequence_options = [(f"Sequence {seq}", seq) for seq in available_sequences]
        
        # Create widgets
        feature_widget = widgets.Dropdown(
            options=available_features,
            value='local_CTCF_mean' if 'local_CTCF_mean' in available_features else available_features[0],
            description='Feature:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        channel_widget = widgets.Dropdown(
            options=channel_options,
            value=available_channel_indices[1] if len(available_channel_indices) > 1 else available_channel_indices[0],
            description='Channel:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        time_points_widget = widgets.SelectMultiple(
            options=sequence_options,
            value=available_sequences[:4],  # Select first 4 by default
            description='Time Points:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px', height='120px')
        )
        
        treatment_widget = widgets.Text(
            value='Histogram',
            description='Name:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        x_scale_widget = widgets.ToggleButtons(
            options=[('Linear', 'linear'), ('Log', 'log'), ('SymLog', 'symlog')],
            value='log',
            description='X Scale:',
            style={'description_width': 'initial'}
        )
        
        linthresh_widget = widgets.FloatText(
            value=1.0,
            description='LinThresh:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        n_bins_widget = widgets.IntSlider(
            value=50,
            min=10,
            max=200,
            step=10,
            description='N Bins:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        alpha_widget = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Alpha:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        cumulative_widget = widgets.Checkbox(
            value=False,
            description='Cumulative',
            style={'description_width': 'initial'}
        )
        
        outline_widget = widgets.Checkbox(
            value=True,
            description='Outline',
            style={'description_width': 'initial'}
        )
        
        smooth_widget = widgets.Checkbox(
            value=True,
            description='Smooth',
            style={'description_width': 'initial'}
        )
        
        constant_n_widget = widgets.Checkbox(
            value=True,
            description='Constant N',
            style={'description_width': 'initial'}
        )
        
        hide_negative_ctcf_widget = widgets.Checkbox(
            value=True,
            description='Show Positive CTCF only',
            style={'description_width': 'initial'}
        )
        
        border_width_widget = widgets.FloatSlider(
            value=1.5,
            min=0.5,
            max=5.0,
            step=0.5,
            description='Border Width:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        font_size_widget = widgets.IntSlider(
            value=11,
            min=8,
            max=20,
            step=1,
            description='Font Size:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Show/hide linthresh widget based on x_scale selection
        def update_linthresh_visibility(change):
            linthresh_widget.layout.display = 'block' if change['new'] == 'symlog' else 'none'
        
        x_scale_widget.observe(update_linthresh_visibility, names='value')
        linthresh_widget.layout.display = 'block' if x_scale_widget.value == 'symlog' else 'none'
        
        # Changed button text and behavior - only save parameters, no auto-plot
        save_button = widgets.Button(
            description='💾 Save Parameters',
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        output_area = widgets.Output()
        
        def on_save_click(button):
            """Save the parameters without generating a plot."""
            with output_area:
                output_area.clear_output()
                
                selected_feature = feature_widget.value
                selected_channel = channel_widget.value
                selected_time_points = list(time_points_widget.value)
                selected_treatment = treatment_widget.value
                selected_x_scale = x_scale_widget.value
                selected_linthresh = linthresh_widget.value
                selected_n_bins = n_bins_widget.value
                selected_alpha = alpha_widget.value
                selected_cumulative = cumulative_widget.value
                selected_outline = outline_widget.value
                selected_smooth = smooth_widget.value
                selected_constant_n = constant_n_widget.value
                selected_hide_negative = hide_negative_ctcf_widget.value
                selected_border_width = border_width_widget.value
                selected_font_size = font_size_widget.value
                
                if not selected_time_points:
                    print("❌ Please select at least one time point")
                    return
                
                # Get channel name for display
                channel_name = channel_mapping.get(selected_channel, f"Channel {selected_channel}")
                
                print(f"💾 Parameters saved successfully:")
                print(f"   Feature: {selected_feature}")
                print(f"   Channel: {channel_name}") 
                print(f"   Time Points: {[int(tp) for tp in selected_time_points]}")
                print(f"   Name: {selected_treatment}")
                print(f"   X Scale: {selected_x_scale}")
                if selected_x_scale == 'symlog':
                    print(f"   LinThresh: {selected_linthresh}")
                print(f"   N Bins: {selected_n_bins}")
                print(f"   Alpha: {selected_alpha}")
                print(f"   Options: Cumulative={selected_cumulative}, Outline={selected_outline}, Smooth={selected_smooth}")
                print(f"   Advanced: Constant N={selected_constant_n}, Show Positive CTCF Only={selected_hide_negative}")
                print(f"   Border Width: {selected_border_width}, Font Size: {selected_font_size}")
                print(f"✅ Use the next cell to generate the plot with these parameters.")
        
        save_button.on_click(on_save_click)
        
        # Display the controls
        controls_layout = widgets.VBox([
            widgets.HTML("<h3>📊 Multi-Timepoint Histogram Configuration</h3>"),
            widgets.HTML("<p>Configure parameters for plotting feature histograms across multiple time points:</p>"),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<b>Data Selection:</b>"),
                    feature_widget,
                    channel_widget,
                    time_points_widget,
                    treatment_widget
                ], layout=widgets.Layout(margin='0px 20px 0px 0px')),
                widgets.VBox([
                    widgets.HTML("<b>Scale & Bins:</b>"),
                    x_scale_widget,
                    linthresh_widget,
                    n_bins_widget,
                    alpha_widget,
                    border_width_widget,
                    font_size_widget
                ], layout=widgets.Layout(margin='0px 20px 0px 0px')),
                widgets.VBox([
                    widgets.HTML("<b>Options:</b>"),
                    cumulative_widget,
                    outline_widget,
                    smooth_widget,
                    constant_n_widget,
                    hide_negative_ctcf_widget
                ])
            ]),
            save_button,
            output_area
        ])
        
        display(controls_layout)
        
        # Store references for external access
        controls_layout.feature_widget = feature_widget
        controls_layout.channel_widget = channel_widget
        controls_layout.time_points_widget = time_points_widget
        controls_layout.treatment_widget = treatment_widget
        controls_layout.x_scale_widget = x_scale_widget
        controls_layout.linthresh_widget = linthresh_widget
        controls_layout.n_bins_widget = n_bins_widget
        controls_layout.alpha_widget = alpha_widget
        controls_layout.cumulative_widget = cumulative_widget
        controls_layout.outline_widget = outline_widget
        controls_layout.smooth_widget = smooth_widget
        controls_layout.constant_n_widget = constant_n_widget
        controls_layout.hide_negative_ctcf_widget = hide_negative_ctcf_widget
        controls_layout.border_width_widget = border_width_widget
        controls_layout.font_size_widget = font_size_widget
        controls_layout.output_area = output_area
        
        return controls_layout
        
    except Exception as e:
        print(f"❌ Error creating histogram controls: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_histogram_from_controls(filtered_df, sample, controls_layout):
    """
    Generate a multi-timepoint histogram using current values from controls_layout.
    """
    if controls_layout and hasattr(controls_layout, 'feature_widget'):
        plot_multi_timepoint_hist(
            filtered_df,
            channel_mapping=sample.channels,
            feature_name=controls_layout.feature_widget.value,
            selected_time_points=list(controls_layout.time_points_widget.value),
            channel=controls_layout.channel_widget.value,
            treatment=controls_layout.treatment_widget.value,
            time_column='sequence',
            x_scale=controls_layout.x_scale_widget.value,
            linthresh=controls_layout.linthresh_widget.value,
            n_bins=controls_layout.n_bins_widget.value,
            font_size=controls_layout.font_size_widget.value,
            alpha=controls_layout.alpha_widget.value,
            cumulative=controls_layout.cumulative_widget.value,
            outline=controls_layout.outline_widget.value,
            smooth=controls_layout.smooth_widget.value,
            border_width=controls_layout.border_width_widget.value,
            constant_N=controls_layout.constant_n_widget.value,
            random_seed=42,
            hide_negative_ctcf=controls_layout.hide_negative_ctcf_widget.value
        )
    else:
        print("❌ Histogram controls not found. Please run the histogram controls cell above first.")


def create_time_series_controls(df, sample, data_type="wells_data"):
    """Create interactive controls for time series plotting with enhanced button behavior."""
    try:
        # Get available features from the data
        available_features = [col for col in df.columns if col in PARQUET_FILES[data_type]["feature_columns"]]
        if not available_features:
            print("❌ No feature columns available in the data")
            return
        
        # Get available channels from the data and sample
        available_channel_indices = sorted(df['channel_index'].unique()) if 'channel_index' in df.columns else []
        channel_mapping = sample.channels
        
        # Get sequence range from data
        available_sequences = sorted(df['sequence'].unique()) if 'sequence' in df.columns else [0, 1, 2, 3, 4, 5]
        min_seq, max_seq = min(available_sequences), max(available_sequences)
        
        # Create channel options with names
        channel_options = []
        for idx in available_channel_indices:
            channel_name = channel_mapping.get(idx, f"Channel {idx}")
            channel_options.append((f"{idx}: {channel_name}", idx))
        
        # Create widgets
        feature_widget = widgets.Dropdown(
            options=available_features,
            value='well_intensity_mean' if 'well_intensity_mean' in available_features else available_features[0],
            description='Feature:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        channels_widget = widgets.SelectMultiple(
            options=channel_options,
            value=available_channel_indices,  # Select all channels by default
            description='Channels:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px', height='120px')
        )
        
        # Sequence range widgets
        start_sequence_widget = widgets.IntSlider(
            value=min_seq,
            min=min_seq,
            max=max_seq,
            step=1,
            description='Start Seq:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        end_sequence_widget = widgets.IntSlider(
            value=max_seq,
            min=min_seq,
            max=max_seq,
            step=1,
            description='End Seq:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        y_scale_widget = widgets.ToggleButtons(
            options=[('Linear', 'linear'), ('Log', 'log')],
            value='linear',
            description='Y Scale:',
            style={'description_width': 'initial'}
        )
        
        error_bar_widget = widgets.ToggleButtons(
            options=[('Standard Error (SEM)', 'sem'), ('Standard Deviation', 'std')],
            value='sem',
            description='Error Bars:',
            style={'description_width': 'initial'}
        )
        
        show_error_bars_widget = widgets.Checkbox(
            value=True,
            description='Show Error Bars',
            style={'description_width': 'initial'}
        )
        
        filter_non_positive_widget = widgets.Checkbox(
            value=False,
            description='Filter Non-Positive (for log scale)',
            style={'description_width': 'initial'}
        )
        
        # Changed button text and behavior - only save parameters, no auto-plot
        save_button = widgets.Button(
            description='💾 Save Parameters',
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        output_area = widgets.Output()
        
        # Function to validate sequence range
        def validate_sequence_range():
            """Ensure start <= end sequence."""
            if start_sequence_widget.value > end_sequence_widget.value:
                end_sequence_widget.value = start_sequence_widget.value
        
        def on_start_sequence_change(change):
            validate_sequence_range()
            
        def on_end_sequence_change(change):
            if end_sequence_widget.value < start_sequence_widget.value:
                start_sequence_widget.value = end_sequence_widget.value
        
        start_sequence_widget.observe(on_start_sequence_change, names='value')
        end_sequence_widget.observe(on_end_sequence_change, names='value')
        
        def on_save_click(button):
            """Save the parameters without generating a plot."""
            with output_area:
                output_area.clear_output()
                
                selected_feature = feature_widget.value
                selected_channels = list(channels_widget.value)
                selected_start_seq = start_sequence_widget.value
                selected_end_seq = end_sequence_widget.value
                selected_y_scale = y_scale_widget.value
                selected_error_bar = error_bar_widget.value
                show_errors = show_error_bars_widget.value
                filter_non_pos = filter_non_positive_widget.value
                
                if not selected_channels:
                    print("❌ Please select at least one channel")
                    return
                
                # Get channel names for display
                channel_names = []
                for ch in selected_channels:
                    channel_name = channel_mapping.get(ch, f"Channel {ch}")
                    channel_names.append(f"{ch}: {channel_name}")
                
                print(f"💾 Parameters saved successfully:")
                print(f"   Feature: {selected_feature}")
                print(f"   Channels: {channel_names}")  # Show channel names instead of numbers
                print(f"   Sequence Range: {selected_start_seq} - {selected_end_seq}")
                print(f"   Y Scale: {selected_y_scale}")
                print(f"   Error Bars: {selected_error_bar} ({'shown' if show_errors else 'hidden'})")
                print(f"   Filter Non-Positive: {filter_non_pos}")
                print(f"✅ Use the next cell to generate the plot with these parameters.")
        
        save_button.on_click(on_save_click)
        
        # Display the controls
        controls_layout = widgets.VBox([
            widgets.HTML("<h3>📈 Time Series Plot Configuration</h3>"),
            widgets.HTML("<p>Configure parameters for plotting feature values over time by channel:</p>"),
            widgets.HBox([
                widgets.VBox([
                    feature_widget,
                    channels_widget
                ], layout=widgets.Layout(margin='0px 20px 0px 0px')),
                widgets.VBox([
                    start_sequence_widget,
                    end_sequence_widget,
                    y_scale_widget,
                    error_bar_widget
                ], layout=widgets.Layout(margin='0px 20px 0px 0px')),
                widgets.VBox([
                    show_error_bars_widget,
                    filter_non_positive_widget
                ])
            ]),
            save_button,
            output_area
        ])
        
        display(controls_layout)
        
        # Store references for external access
        controls_layout.feature_widget = feature_widget
        controls_layout.channels_widget = channels_widget
        controls_layout.start_sequence_widget = start_sequence_widget
        controls_layout.end_sequence_widget = end_sequence_widget
        controls_layout.y_scale_widget = y_scale_widget
        controls_layout.error_bar_widget = error_bar_widget
        controls_layout.show_error_bars_widget = show_error_bars_widget
        controls_layout.filter_non_positive_widget = filter_non_positive_widget
        controls_layout.output_area = output_area
        
        return controls_layout
        
    except Exception as e:
        print(f"❌ Error creating time series controls: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_feature_channels_over_time(
    df,
    feature_name,
    channel_mapping,
    start_sequence=None,
    end_sequence=None,
    y_scale='linear',            # 'linear' or 'log'
    error_bar='sem',             # 'sem' or 'std'
    show_error_bars=True,
    channels=None,               # list of channel indices (e.g., [0,1,2])
    filter_non_positive=False
):
    """
    Plots mean ± error bars of a feature over sequences for different channels using Bokeh.
    Hover shows info for lines/points only (no hover tooltips for error bars).
    """

    # --- Filter sequence range
    if start_sequence is not None:
        df = df[df['sequence'] >= start_sequence]
    if end_sequence is not None:
        df = df[df['sequence'] <= end_sequence]

    # --- Channels to use
    available_channels = sorted(df['channel_index'].dropna().unique().tolist())
    if channels is None:
        channels = available_channels
    else:
        df = df[df['channel_index'].isin(channels)]
        channels = sorted(channels)

    # --- Prepare sequences
    df = df.sort_values(by='sequence')
    sequences = sorted(df['sequence'].dropna().unique().tolist())

    # --- Choose a palette that won't KeyError for mid sizes
    def pick_colors(n):
        if n <= 3:
            base = Category10[3]
        elif n <= 10:
            base = Category10[10]
        elif n <= 20:
            base = Category20[20]
        else:
            # sample n roughly-evenly spaced colors from Turbo256
            step = max(1, len(Turbo256)//n)
            base = list(islice(Turbo256, 0, step*n, step))
        return base[:n]

    colors = pick_colors(len(channels))

    # --- Figure (set y axis type here so log works)
    y_axis_type = 'log' if y_scale.lower() == 'log' else 'linear'
    p = figure(
        width=950,
        height=420,
        tools=[],  # add explicitly below
        y_axis_type=y_axis_type,
        title=f'{feature_name.replace("_", " ").title()} over sequences by channel'
    )

    # Add standard tools
    p.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), SaveTool())

    # We'll attach the hover to these renderers only (not to error bars)
    hover_renderers = []

    # --- Plot per channel
    for color, ch in zip(colors, channels):
        means, errors, seq_list = [], [], []

        ch_df = df[df['channel_index'] == ch]
        # Iterate through all sequences in range so x aligns across channels
        for seq in sequences:
            vals = ch_df.loc[ch_df['sequence'] == seq, feature_name].dropna()

            if filter_non_positive:
                vals = vals[vals > 0]

            if len(vals) == 0:
                means.append(np.nan)
                errors.append(0.0)
            else:
                mean_val = float(vals.mean())
                if show_error_bars:
                    if error_bar == 'sem':
                        err = float(vals.sem()) if len(vals) > 1 else 0.0
                    elif error_bar == 'std':
                        err = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    else:
                        err = 0.0
                else:
                    err = 0.0

                means.append(mean_val)
                errors.append(err)

            seq_list.append(seq)

        # Build data source (include string label for hover)
        label = channel_mapping.get(ch, f"Ch {ch}")

        # For log scale, ensure error bars don't go <= 0
        if y_axis_type == 'log':
            tiny = 1e-12
            y_lower = [max((m - e) if (m is not None) else np.nan, tiny) if (m is not np.nan) else np.nan
                       for m, e in zip(means, errors)]
        else:
            y_lower = [(m - e) if (m is not np.nan) else np.nan for m, e in zip(means, errors)]

        y_upper = [(m + e) if (m is not np.nan) else np.nan for m, e in zip(means, errors)]

        # Error bars (draw first so points/lines sit on top). No hover bound to these.
        if show_error_bars:
            err_source = ColumnDataSource(dict(
                x=seq_list,
                y_lower=y_lower,
                y_upper=y_upper
            ))
            p.segment('x', 'y_lower', 'x', 'y_upper',
                      source=err_source, line_color=color, line_width=1, alpha=0.8)

        # Line + points
        source = ColumnDataSource(dict(
            x=seq_list,
            y=means,
            error=errors,
            channel_name=[label]*len(seq_list)
        ))
        line_r = p.line('x', 'y', source=source, line_color=color, line_width=2, legend_label=label)
        circ_r = p.scatter('x', 'y', source=source, fill_color=color, line_color=color,
                          size=6, legend_label=label)

        hover_renderers.extend([line_r, circ_r])

    # --- Hover (bind to line+points only, so error bars show nothing)
    hover = HoverTool(
    tooltips=[
        ('Channel', '@channel_name'),
        ('Sequence', '@x'),
        ('Mean', '@y{0.00}'),
        ('Error', '@error{0.00}')
    ],
    renderers=hover_renderers,   # line + circles (no error bars)
    mode='mouse',                # <- was 'vline'
    point_policy='snap_to_data', # snap to nearest point
    line_policy='nearest'        # use nearest vertex on lines
)
    p.add_tools(hover)

    # --- Styling
    p.xaxis.axis_label = 'Sequence'
    p.yaxis.axis_label = feature_name.replace("_", " ")
    p.grid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    show(p)


def plot_time_series_from_controls(df, sample, controls_layout):
    """
    Generate a time series plot using current values from controls_layout.
    """
    if controls_layout and hasattr(controls_layout, 'feature_widget'):
        plot_feature_channels_over_time(
            df,
            feature_name=controls_layout.feature_widget.value,
            channel_mapping=sample.channels,
            start_sequence=controls_layout.start_sequence_widget.value,
            end_sequence=controls_layout.end_sequence_widget.value,
            y_scale=controls_layout.y_scale_widget.value,
            error_bar=controls_layout.error_bar_widget.value,
            show_error_bars=controls_layout.show_error_bars_widget.value,
            channels=list(controls_layout.channels_widget.value),
            filter_non_positive=controls_layout.filter_non_positive_widget.value
        )
    else:
        print("❌ Controls not found. Please run the controls cell above first.")
