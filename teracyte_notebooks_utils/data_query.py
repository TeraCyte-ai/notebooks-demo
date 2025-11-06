"""
Data query utilities for TeraCyte data overview notebooks.

This module provides utilities for reading experimental data.
"""

# Standard library imports
import io
import os
import traceback
import warnings
from typing import Dict, Union, List

# Third-party imports
import duckdb
import fsspec
import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display, HTML
from ipywidgets import FileUpload, Button, VBox, HTML as HTMLWidget, Output

# Local imports
from .sample import Sample
from .api_utils import get_assay_workflows

# Suppress specific pandas FutureWarnings about groupby operations
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns", category=FutureWarning)


def query_filtered_parquet_from_azure(
    parquet_data_path: str,
    account_name: str,
    sas_token: str,
    filters: Dict[str, Union[str, int, float, List]],
    columns: List[str] = None,
) -> pd.DataFrame:
    """
    Query filtered rows from a partitioned Parquet dataset on Azure Blob Storage using DuckDB.

    Parameters:
    - parquet_data_path (str): Path to the folder with Parquet partitions in Azure (not including container).
    - account_name (str): Azure Blob Storage account name.
    - sas_token (str): SAS token with read access.
    - filters (dict): Dictionary of partition column filters. Values can be single values or lists.
                      (e.g., {'sequence': 0, 'fov': [1, 2, 3], 'channel_index': [0, 1]}).
    - columns (list, optional): List of column names to select. If None, selects all columns.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    
    Example:
        >>> filters = {"sequence": 0, "fov": [0, 1], "z_index": 0, "channel_index": [0, 1]}
        >>> columns = ["fov", "sequence", "channel_index", "mean_intensity"]
        >>> df = read_filtered_parquet_from_azure(
        ...     parquet_data_path="user/assay123/wells_data.parquet",
        ...     account_name="mystorageaccount",
        ...     sas_token="sp=r&st=2024...",
        ...     filters=filters,
        ...     columns=columns
        ... )
    """
    # Initialize Azure fsspec filesystem
    fs = fsspec.filesystem("az", account_name=account_name, sas_token=sas_token)

    # List only valid parquet data files (skip metadata)
    parquet_urls = [
        f"https://{account_name}.blob.core.windows.net/{path}?{sas_token}"
        for path in fs.find(parquet_data_path)
        if path.endswith(".parquet") and "/_" not in path
    ]

    if not parquet_urls:
        raise RuntimeError("No .parquet data files found at the given path.")

    # Connect to DuckDB and enable HTTP access
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs")

    # Build WHERE clause from filters
    where_conditions = []
    for k, v in filters.items():
        if isinstance(v, list):
            # Handle list of values using IN clause
            if all(isinstance(item, str) for item in v):
                # String values need quotes
                values_str = "', '".join(str(item) for item in v)
                where_conditions.append(f"{k} IN ('{values_str}')")
            else:
                # Numeric values don't need quotes
                values_str = ", ".join(str(item) for item in v)
                where_conditions.append(f"{k} IN ({values_str})")
        else:
            # Handle single value
            if isinstance(v, str):
                where_conditions.append(f"{k} = '{v}'")
            else:
                where_conditions.append(f"{k} = {v}")

    # Build SELECT clause
    if columns:
        # Validate that requested columns exist or are partition columns
        select_clause = ", ".join(columns)
    else:
        select_clause = "*"

    # Build final query - only add WHERE clause if we have conditions
    if where_conditions:
        where_clause = " AND ".join(where_conditions)
        query = f"""
        SELECT {select_clause} FROM read_parquet({parquet_urls}, hive_partitioning=true)
        WHERE {where_clause}
        """
    else:
        # No filters - select all data
        query = f"""
        SELECT {select_clause} FROM read_parquet({parquet_urls}, hive_partitioning=true)
        """

    # Execute query and return as DataFrame
    return con.execute(query).fetchdf()


def read_sample_parquet_data(
    sample: Sample,
    filters: Dict[str, Union[str, int, float]],
    columns: List[str] = None,
    parquet_type: str = "wells_data"
) -> pd.DataFrame:
    """
    Convenience function to read filtered parquet data using a Sample object.
    
    Parameters:
    - sample (Sample): Sample object containing Azure configuration and paths
    - filters (dict): Dictionary of partition column filters
    - columns (list, optional): List of column names to select. If None, selects all columns.
    - parquet_type (str): Type of parquet file to query (default: "wells_data")
    
    Returns:
    - pd.DataFrame: Filtered DataFrame
    
    Example:
        >>> sample = Sample("serial123", "exp456", "assay789")
        >>> filters = {"sequence": 0, "fov": 0}
        >>> columns = ["fov", "sequence", "mean_intensity"]
        >>> df = read_sample_parquet_data(sample, filters, columns, parquet_type="cells_data")
    """    
    # Check if the parquet type is available
    if not sample.has_parquet_type(parquet_type):
        raise ValueError(f"Data type '{parquet_type}' is not available for this sample. Available types: {sample.available_parquet_types}")
    
    metadata = sample.get_parquet_metadata(parquet_type)
    parquet_data_path = metadata.get("path", "")
    
    return query_filtered_parquet_from_azure(
        parquet_data_path=parquet_data_path,
        account_name=sample.account_name,
        sas_token=sample.sas,
        filters=filters,
        columns=columns
    )


def create_interactive_data_query(sample: Sample, always_checked_columns: Dict = {'fov', 'sequence', 'timepoint', 'channel_index', 'z_index', 'classification', 'global_index', 'object_id', 'channel_index1', 'channel_index2'}) -> widgets.VBox:
    """
    Create an interactive query builder with parquet type selection that displays results without saving to sample.
    
    Parameters:
    - sample (Sample): Sample object containing metadata and configuration
    - always_checked_columns (set): Set of column names to check by default
    
    Returns:
    - widgets.VBox: Interactive widget for data querying
    """
    # Check if sample has any parquet files available
    available_types = sample.available_parquet_types
    if not available_types:
        return widgets.HTML(f"<div style='color: red;'>Error: No data files available for this sample</div>")
    
    # Create parquet type selection dropdown
    parquet_type_dropdown = widgets.Dropdown(
        options=[(f"{ptype} - {sample.get_parquet_metadata(ptype).get('description', 'No description')}", ptype) 
                for ptype in available_types],
        value=available_types[0] if available_types else None,
        description='Data type:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    
    # Create containers for dynamic content that will update based on parquet type selection
    dynamic_content_area = widgets.VBox()
    
    # Variables to store current widgets (will be updated when parquet type changes)
    current_widgets = {
        'column_checkboxes': [],
        'select_all_columns': None,
        'partition_widgets': {},
        'summary_area': None,
        'available_columns': [],
        'total_rows': 0,
        'load_button': None,
        'output_area': None
    }
    
    def build_widgets_for_parquet_type(parquet_type):
        """Build all widgets for a specific parquet type."""
        # Get metadata for this parquet type
        metadata = sample.get_parquet_metadata(parquet_type)
        if "error" in metadata:
            error_widget = widgets.HTML(f"<div style='color: red;'>Error loading {parquet_type}: {metadata['error']}</div>")
            return error_widget
        
        # Get available columns from parquet metadata
        available_columns = metadata.get('columns', [])
        
        if not available_columns:
            error_widget = widgets.HTML(f"<div style='color: red;'>No column information available for {parquet_type}</div>")
            return error_widget
        
        # Store available columns for later use
        current_widgets['available_columns'] = available_columns
        
        # Get total number of rows from metadata
        total_rows = metadata.get('num_rows', 0)
        current_widgets['total_rows'] = total_rows
        
        # Get channel names from sample
        channel_names = sample.channels
        
        # Create column selection widgets
        column_checkboxes = []
        for col in available_columns:
            # Check if this column should be checked by default
            default_checked = col in always_checked_columns
            
            checkbox = widgets.Checkbox(
                value=default_checked,
                description=col,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px', margin='0px 5px 5px 0px')
            )
            column_checkboxes.append(checkbox)
        
        current_widgets['column_checkboxes'] = column_checkboxes
        
        # Create "Select All Columns" checkbox
        all_columns_checked_by_default = all(col in always_checked_columns for col in available_columns)
        select_all_columns_checkbox = widgets.Checkbox(
            value=all_columns_checked_by_default,
            description="Select All Columns",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        current_widgets['select_all_columns'] = select_all_columns_checkbox
        
        # Function to handle "Select All Columns" behavior
        def handle_select_all_columns(change):
            for checkbox in current_widgets['column_checkboxes']:
                checkbox.value = change['new']
        
        select_all_columns_checkbox.observe(handle_select_all_columns, names='value')
        
        def handle_individual_column_change(change):
            all_selected = all(cb.value for cb in current_widgets['column_checkboxes'])
            none_selected = not any(cb.value for cb in current_widgets['column_checkboxes'])
            if all_selected:
                current_widgets['select_all_columns'].value = True
            elif none_selected:
                current_widgets['select_all_columns'].value = False
        
        for checkbox in column_checkboxes:
            checkbox.observe(handle_individual_column_change, names='value')
        
        # Create scrollable container for column checkboxes
        columns_container = widgets.VBox(
            column_checkboxes,
            layout=widgets.Layout(
                height='200px',
                overflow_y='auto',
                border='1px solid #ddd',
                padding='5px',
                margin='5px 0px'
            )
        )
        
        # Create the complete column selection widget
        column_selection_widget = widgets.VBox([
            widgets.HTML(f"<b>Columns</b> ({len(available_columns)} available)"),
            select_all_columns_checkbox,
            columns_container
        ], layout=widgets.Layout(width='250px', margin='0px 15px 0px 0px'))
        
        # Create partition widgets
        partition_widgets = {}
        partition_values = metadata.get('partition_values', {})
        
        for partition_name, values in partition_values.items():
            # Create "Select All" checkbox
            select_all_checkbox = widgets.Checkbox(
                value=False,
                description="Select All",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='120px')
            )
            
            # Create individual value checkboxes
            value_checkboxes = []
            for value in values:
                if partition_name == 'channel_index' and value in channel_names:
                    display_name = f"{value}: {channel_names[value]}"
                else:
                    display_name = str(value)
                
                checkbox = widgets.Checkbox(
                    value=False,
                    description=display_name,
                    style={'description_width': 'initial'},
                    layout=widgets.Layout(width='120px', margin='0px 5px 0px 0px')
                )
                value_checkboxes.append(checkbox)
            
            # Create scrollable container for value checkboxes
            checkbox_container = widgets.VBox(
                value_checkboxes,
                layout=widgets.Layout(
                    height='100px',
                    overflow_y='auto',
                    border='1px solid #ddd',
                    padding='5px',
                    margin='5px 0px'
                )
            )
            
            # Function to handle "Select All" behavior
            def create_select_all_handler(checkboxes, select_all):
                def handle_select_all(change):
                    for checkbox in checkboxes:
                        checkbox.value = change['new']
                select_all.observe(handle_select_all, names='value')
                
                def handle_individual_change(change):
                    all_selected = all(cb.value for cb in checkboxes)
                    none_selected = not any(cb.value for cb in checkboxes)
                    if all_selected:
                        select_all.value = True
                    elif none_selected:
                        select_all.value = False
                
                for checkbox in checkboxes:
                    checkbox.observe(handle_individual_change, names='value')
            
            # Set up the select all handler
            create_select_all_handler(value_checkboxes, select_all_checkbox)
            
            # Create the complete widget for this partition
            partition_widget = widgets.VBox([
                widgets.HTML(f"<b>{partition_name}</b> ({len(values)} values)"),
                select_all_checkbox,
                checkbox_container
            ], layout=widgets.Layout(width='200px', margin='0px 10px 0px 0px'))
            
            partition_widgets[partition_name] = {
                'container': partition_widget,
                'select_all': select_all_checkbox,
                'checkboxes': value_checkboxes,
                'values': values
            }
        
        current_widgets['partition_widgets'] = partition_widgets
        
        # Create summary area
        summary_area = widgets.HTML(
            value="<div style='padding: 10px; background: #fff3cd; border-radius: 5px; color: #856404;'>Select partition values to enable data loading.</div>"
        )
        current_widgets['summary_area'] = summary_area
        
        # Create load button
        load_button = widgets.Button(
            description='Load Data',
            button_style='primary',
            layout=widgets.Layout(width='150px', height='40px')
        )
        current_widgets['load_button'] = load_button
        
        # Create clear button for this parquet type
        clear_button = widgets.Button(
            description=f'Clear {parquet_type}',
            button_style='warning',
            layout=widgets.Layout(width='150px', height='40px')
        )
        
        def clear_current_dataframe(button):
            """Clear the dataframe for the current parquet type."""
            sample.clear_dataframe(parquet_type)
            update_dataframe_status_display()
            with current_widgets['output_area']:
                clear_output(wait=True)
                print(f"🗑️ Cleared DataFrame for {parquet_type}")
        
        clear_button.on_click(clear_current_dataframe)
        
        # Create output area for results
        output_area = widgets.Output()
        current_widgets['output_area'] = output_area
        
        def update_summary():
            """Update the summary based on current selections."""
            filters = {}
            total_combinations = 1
            
            # Count selected columns
            selected_columns = sum(1 for cb in current_widgets['column_checkboxes'] if cb.value)
            
            for partition_name, widget_info in current_widgets['partition_widgets'].items():
                selected_values = []
                for i, checkbox in enumerate(widget_info['checkboxes']):
                    if checkbox.value:
                        selected_values.append(widget_info['values'][i])
                
                if selected_values:
                    if len(selected_values) == len(widget_info['values']):
                        total_combinations *= len(widget_info['values'])
                    else:
                        filters[partition_name] = selected_values
                        total_combinations *= len(selected_values)
                else:
                    total_combinations = 0
            
            # Add total rows info to all summary messages
            total_rows_info = f"<br><strong>Total rows in data file:</strong> {current_widgets['total_rows']:,}"
            
            if total_combinations == 0:
                summary_text = f"<div style='padding: 10px; background: #fff3cd; border-radius: 5px; color: #856404;'><strong>No data will be returned</strong> - At least one value must be selected for each partition.{total_rows_info}</div>"
            elif selected_columns == 0:
                summary_text = f"<div style='padding: 10px; background: #fff3cd; border-radius: 5px; color: #856404;'><strong>No columns selected</strong> - At least one column must be selected.{total_rows_info}</div>"
            elif filters:
                filter_text = ", ".join([f"{k}: {len(v)} values" for k, v in filters.items()])
                summary_text = f"<div style='padding: 10px; background: #e7f3ff; border-radius: 5px;'><strong>Columns:</strong> {selected_columns}/{len(current_widgets['available_columns'])}<br><strong>Filters:</strong> {filter_text}<br><strong>Estimated combinations:</strong> {total_combinations:,}{total_rows_info}</div>"
            else:
                summary_text = f"<div style='padding: 10px; background: #fff3cd; border-radius: 5px;'><strong>Columns:</strong> {selected_columns}/{len(current_widgets['available_columns'])}<br><strong>All partitions selected</strong><br><strong>Estimated combinations:</strong> {total_combinations:,} (this may take time!){total_rows_info}</div>"
            
            current_widgets['summary_area'].value = summary_text
        
        def on_checkbox_change(change):
            """Handle checkbox changes."""
            update_summary()
        
        # Connect change handlers to all checkboxes (columns and partitions)
        select_all_columns_checkbox.observe(on_checkbox_change, names='value')
        for checkbox in column_checkboxes:
            checkbox.observe(on_checkbox_change, names='value')
            
        for widget_info in partition_widgets.values():
            widget_info['select_all'].observe(on_checkbox_change, names='value')
            for checkbox in widget_info['checkboxes']:
                checkbox.observe(on_checkbox_change, names='value')
        
        def load_data_with_filters(button):
            """Load data based on selected filters and display table."""
            with current_widgets['output_area']:
                clear_output(wait=True)
                print("🔄 Building query...")
                
                # Get selected columns
                selected_columns = []
                for i, checkbox in enumerate(current_widgets['column_checkboxes']):
                    if checkbox.value:
                        selected_columns.append(current_widgets['available_columns'][i])
                
                if not selected_columns:
                    print("❌ No columns selected")
                    print("💡 Please select at least one column to load.")
                    return
                
                # Build filters dictionary
                filters = {}
                has_empty_partition = False
                
                for partition_name, widget_info in current_widgets['partition_widgets'].items():
                    selected_values = []
                    for i, checkbox in enumerate(widget_info['checkboxes']):
                        if checkbox.value:
                            selected_values.append(widget_info['values'][i])
                    
                    if not selected_values:
                        print(f"❌ No values selected for partition '{partition_name}'")
                        has_empty_partition = True
                    elif len(selected_values) < len(widget_info['values']):
                        if len(selected_values) == 1:
                            filters[partition_name] = selected_values[0]
                        else:
                            filters[partition_name] = selected_values
                
                if has_empty_partition:
                    print("💡 Please select at least one value for each partition.")
                    return
                
                print(f"📋 Data type: {parquet_type}")
                print(f"📋 Selected columns: {selected_columns}")
                print(f"📋 Query filters: {filters}")
                
                try:
                    print("🚀 Executing DuckDB query...")
                    df = read_sample_parquet_data(sample, filters, columns=selected_columns, parquet_type=parquet_type)
                    
                    # Calculate percentage of data loaded
                    total_rows_in_parquet = current_widgets['total_rows']
                    loaded_rows = len(df)
                    if total_rows_in_parquet > 0:
                        percentage_loaded = (loaded_rows / total_rows_in_parquet) * 100
                        print(f"✅ Successfully loaded {loaded_rows:,} rows ({percentage_loaded:.2f}% of total {total_rows_in_parquet:,} rows)")
                    else:
                        print(f"✅ Successfully loaded {loaded_rows:,} rows")
                    
                    print(f"📊 Data shape: {df.shape}")
                    print(f"🗂️  Columns: {list(df.columns)}")
                    
                    # Store in sample using new dataframe method
                    sample.set_dataframe(parquet_type, dataframe=df, filters=filters)
                    print(f"💾 Data stored in sample.get_dataframe('{parquet_type}')")
                    print()
                                        
                except Exception as e:
                    print(f"❌ Error loading data: {str(e)}")
                    traceback.print_exc()
            
            # Update dataframe status display (outside the output area)
            update_dataframe_status_display()
        
        # Connect load button
        load_button.on_click(load_data_with_filters)
        
        # Create the layout
        partition_containers = [widget_info['container'] for widget_info in partition_widgets.values()]
        
        # Add column selection to the left of partitions
        all_containers = [column_selection_widget] + partition_containers
        
        selection_row = widgets.HBox(
            all_containers,
            layout=widgets.Layout(
                overflow_x='auto',
                justify_content='flex-start',
                align_items='flex-start'
            )
        )
        
        # Create button row
        button_row = widgets.HBox([
            load_button,
            widgets.HTML("&nbsp;&nbsp;"),  # Spacer
            clear_button
        ], layout=widgets.Layout(justify_content='flex-start'))
        
        controls_section = widgets.VBox([
            widgets.HTML(f"<h4>🔍 Query Configuration for {parquet_type}</h4>"),
            widgets.HTML(f"<p style='color: #666; font-size: 14px;'>Select columns and partition values to query {metadata.get('description', 'data')}. Results are shown below.</p>"),
            selection_row,
            widgets.HTML("<br>"),
            summary_area,
            widgets.HTML("<br>"),
            button_row
        ])
        
        results_section = widgets.VBox([
            widgets.HTML("<h4>📊 Query Results</h4>"),
            output_area
        ])
        
        content_layout = widgets.VBox([
            controls_section,
            widgets.HTML("<hr>"),
            results_section
        ])
        
        # Initialize summary
        update_summary()
        
        return content_layout
    
    def on_parquet_type_change(change):
        """Handle parquet type selection change."""
        selected_type = change['new']
        new_content = build_widgets_for_parquet_type(selected_type)
        dynamic_content_area.children = [new_content]
    
    # Connect the parquet type dropdown to the content update
    parquet_type_dropdown.observe(on_parquet_type_change, names='value')
    
    # Build initial content for the first parquet type
    initial_content = build_widgets_for_parquet_type(available_types[0])
    dynamic_content_area.children = [initial_content]
    
    # Create dataframe status section
    dataframe_status_area = widgets.HTML()
    
    def update_dataframe_status_display():
        """Update the dataframe status display."""
        summary = sample.get_dataframes_summary()
        
        # Create compact horizontal status display
        loaded_items = []
        for ptype, info in summary.items():
            if info['loaded'] and info['shape'][0] > 0:
                rows = f"{info['shape'][0]:,}"
                loaded_items.append(f"✅ <strong>{ptype}</strong>: {rows} rows")
        
        if loaded_items:
            status_html = f"<div style='background: #d4edda; padding: 8px; border-radius: 4px; border-left: 4px solid #28a745; margin: 5px 0;'>"
            status_html += f"📊 <strong>Loaded:</strong> {' | '.join(loaded_items)}"
            status_html += "</div>"
        else:
            status_html = "<div style='background: #fff3cd; padding: 8px; border-radius: 4px; border-left: 4px solid #ffc107; margin: 5px 0;'>📋 No DataFrames loaded</div>"
        
        dataframe_status_area.value = status_html
    
    # Create management buttons
    clear_all_button = widgets.Button(
        description='Clear All DataFrames',
        button_style='warning',
        layout=widgets.Layout(width='200px', height='35px')
    )
    
    def clear_all_dataframes(button):
        """Clear all loaded dataframes."""
        sample.clear_all_dataframes()
        update_dataframe_status_display()
        print("🗑️ All DataFrames cleared")
    
    clear_all_button.on_click(clear_all_dataframes)
    
    # Create the main layout
    main_layout = widgets.VBox([
        widgets.HTML("<h3>🗂️ Data Query</h3>"),
        widgets.HTML("<p style='color: #666; font-size: 14px;'>Select a data type, configure your query, and load the data. Each data type stores its own DataFrame.</p>"),
        dataframe_status_area,
        widgets.HBox([clear_all_button], layout=widgets.Layout(justify_content='flex-end')),
        widgets.HTML("<hr style='margin: 20px 0;'>"),
        parquet_type_dropdown,
        widgets.HTML("<hr style='margin: 20px 0;'>"),
        dynamic_content_area
    ])
    
    # Initialize the dataframe status display
    update_dataframe_status_display()
    
    return main_layout


def download_query_data_csv(sample):
    """
    Downloads CSV files for all loaded DataFrames in the sample.
    Each parquet type gets its own CSV file named after the parquet type.
    
    Args:
        sample: Sample object with loaded DataFrames
    """
    # Get all loaded DataFrames
    summary = sample.get_dataframes_summary()
    
    downloaded_files = []
    for parquet_type, info in summary.items():
        if info['loaded'] and info['shape'][0] > 0:  # Non-empty DataFrame
            df = sample.get_dataframe(parquet_type)
            csv_filename = f"{parquet_type}.csv"
            
            df.to_csv(csv_filename, index=False)
            downloaded_files.append(csv_filename)
            print(f"⬇️ {parquet_type}: {df.shape[0]:,} rows → {csv_filename}")
            
            # Download in Colab if available
            try:
                import google.colab
                from google.colab import files
                files.download(csv_filename)
            except ImportError:
                pass  # Not in Colab
    
    if not downloaded_files:
        print("❌ No loaded DataFrames found - load some data first!")
    else:
        print(f"✅ Downloaded {len(downloaded_files)} CSV files")


def get_assay_workflows_status(assay_id: str) -> pd.DataFrame:
    """
    Get the status of workflows for a specific assay.
    Handles duplicates by returning the last successful workflow for each combination 
    of seq_num, fov_num, and workflow_name, or the last one if all failed.
    """
    workflows = get_assay_workflows(assay_id)
    workflows = workflows.get("workflows")
    if not workflows:
        return pd.DataFrame()
    
    flattened_data = []
    for entry in workflows:
        combined = {**entry, **entry.get('labels', {})}
        combined.pop('labels', None)
        flattened_data.append(combined)

    workflows_df = pd.DataFrame(flattened_data)
    workflows_df = workflows_df[~workflows_df['workflow_name'].isin(['create-assay', 'create-sequence'])]
    workflows_df.reset_index(drop=True, inplace=True)
    workflows_df[['steps_done', 'steps_total']] = workflows_df['progress'].str.split('/', expand=True).astype(int)
    columns_to_drop = [col for col in workflows_df.columns if 'workflows.argoproj.io' in col]
    workflows_df.drop(columns=['progress', 'namespace'] + columns_to_drop, inplace=True)
    workflows_df = workflows_df.astype({
        'assay_num': 'int',
        'seq_num': 'int',
        'fov_num': 'int'
    })
    
    # Handle duplicates: get last successful or last overall for each (seq_num, fov_num, workflow_name)
    if len(workflows_df) > 0:
        # Convert timestamp columns to datetime for proper sorting
        timestamp_cols = ['creation_timestamp', 'started_at', 'finished_at']
        for col in timestamp_cols:
            if col in workflows_df.columns:
                workflows_df[col] = pd.to_datetime(workflows_df[col], errors='coerce')
        
        # Use creation_timestamp as primary sort key (most reliable), fallback to others if needed
        sort_col = 'creation_timestamp'
        if sort_col not in workflows_df.columns or workflows_df[sort_col].isna().all():
            for fallback_col in ['started_at', 'finished_at']:
                if fallback_col in workflows_df.columns and not workflows_df[fallback_col].isna().all():
                    sort_col = fallback_col
                    break
        
        # Sort by timestamp to ensure we get the latest entries
        workflows_df = workflows_df.sort_values(by=sort_col, na_position='last')
        
        # Group by (seq_num, fov_num, workflow_name) and apply deduplication logic
        def get_best_workflow(group):
            # Check if any workflow succeeded
            succeeded = group[group['phase'] == 'Succeeded']
            if len(succeeded) > 0:
                # Return the last successful one (already sorted by timestamp)
                return succeeded.iloc[-1]
            else:
                # No successful workflows, return the last one overall
                return group.iloc[-1]
        
        workflows_df = workflows_df.groupby(['seq_num', 'fov_num', 'workflow_name']).apply(get_best_workflow, include_groups=True).reset_index(drop=True)
    
    return workflows_df


def upload_csv_data():
    """
    Creates an interactive file upload interface that works in both VS Code and Google Colab.
    
    Returns:
        pandas.DataFrame: The uploaded CSV data, or None if no file is selected/uploaded.
        
    Usage:
        data = upload_csv_data()
        if data is not None:
            print(f"Loaded {len(data)} rows")
    """
    # Check if we're in Google Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False
    
    # Storage for the uploaded data
    uploaded_data = {'df': None}
    
    if in_colab:
        # Google Colab implementation
        def upload_colab():
            from google.colab import files
            print("📁 Select your CSV file:")
            uploaded = files.upload()
            
            if uploaded:
                filename = list(uploaded.keys())[0]
                try:
                    df = pd.read_csv(filename)
                    uploaded_data['df'] = df
                    print(f"✅ Successfully loaded {len(df)} rows from {filename}")
                    print(f"Columns: {list(df.columns)}")
                    return df
                except Exception as e:
                    print(f"❌ Error loading CSV: {str(e)}")
                    return None
            else:
                print("❌ No file selected")
                return None
                
        return upload_colab()
        
    else:
        # VS Code/Jupyter implementation with file upload widget
        output = Output()
        
        # Create file upload widget
        file_upload = FileUpload(
            accept='.csv',
            multiple=False,
            description='Upload CSV File',
        )
        
        # Status display
        status_html = HTMLWidget(
            value="<div style='padding: 10px; font-size: 14px;'>📁 Select a CSV file to upload</div>"
        )
        
        def on_upload_change(change):
            """Handle file upload"""
            with output:
                output.clear_output(wait=True)
                
                if file_upload.value:
                    try:
                        # Handle different possible structures of file_upload.value
                        files = file_upload.value
                        
                        if isinstance(files, (tuple, list)) and len(files) > 0:
                            # Files is a tuple/list of file dictionaries
                            file_info = files[0]
                            filename = file_info['name']
                            content = file_info['content']
                        elif isinstance(files, dict):
                            # Files is a single dictionary (alternative structure)
                            filename = files['name']
                            content = files['content']
                        else:
                            raise ValueError("Unexpected file upload structure")
                        
                        # Read the CSV data
                        df = pd.read_csv(io.BytesIO(content))
                        uploaded_data['df'] = df
                        
                        # Update status
                        status_html.value = f"""
                        <div style='padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;'>
                            ✅ Successfully loaded <strong>{len(df)} rows</strong> from <strong>{filename}</strong><br>
                            📊 Columns ({len(df.columns)}): {', '.join(list(df.columns)[:5])}{'...' if len(df.columns) > 5 else ''}
                        </div>
                        """
                                                
                    except Exception as e:
                        status_html.value = f"""
                        <div style='padding: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;'>
                            ❌ Error loading CSV: {str(e)}
                        </div>
                        """
                        print(f"Error: {str(e)}")
                else:
                    status_html.value = "<div style='padding: 10px; font-size: 14px;'>📁 Select a CSV file to upload</div>"
        
        # Connect the upload handler
        file_upload.observe(on_upload_change, names='value')
        
        # Create the UI layout        
        ui = VBox([
            file_upload,
            status_html,
            output
        ])
        
        # Display the interface
        display(ui)
        
        # Return a function to get the data when called
        def get_uploaded_data():
            return uploaded_data['df']
        
        # Store reference for access
        upload_csv_data._get_data = get_uploaded_data
        
        # Return the current data (None initially)
        return uploaded_data['df']


def get_uploaded_data():
    """
    Get the last uploaded CSV data from upload_csv_data() without any display output.
    
    Returns:
        pandas.DataFrame or None: The uploaded data if available, None otherwise
    """
    if hasattr(upload_csv_data, '_get_data'):
        return upload_csv_data._get_data()
    return None


def display_uploaded_data():
    """
    Display information about the last uploaded CSV data from upload_csv_data().
    
    Returns:
        pandas.DataFrame or None: The uploaded data if available, None otherwise
    """
    data = get_uploaded_data()
    
    if data is not None:
        print(f"📊 Previously uploaded data: {data.shape}")
        print(f"Columns: {list(data.columns)}")
    else:
        print("ℹ️ No data has been uploaded yet. Use upload_csv_data() first.")
    
    return data