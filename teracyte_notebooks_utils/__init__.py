"""
TeraCyte Notebooks Utils

A Python SDK for TeraCyte data analysis notebooks, providing utilities for
data access, visualization, and analysis of time-resolved single-cell imaging datasets.
"""

from .sample import Sample
from .constants import (
    CHANNEL_NAME_DICT,
    CHANNEL_COLORS_DICT,
    PARQUET_FILES,
    PARQUET_NON_FEATURE_COLS,
    MAGNIFICATION_GRID_CONFIG
)
from .config import (
    set_service_ip,
    get_service_ip,
    is_configured,
    config
)
from .api_utils import (
    get_experiment_metadata,
    get_sample_metadata,
    get_sample_datapath,
    get_sample_records,
    generate_token,
    get_wells_groups,
    save_wells_group,
    remove_wells_group,
    get_assay_workflows,
)
from .metadata_display import (
    display_hardware_metadata,
    display_sample_metadata,
    display_serial_number_records
)
from .vizarr_viewer import (
    create_fovs_vizarr_viewer
)
from .data_query import (
    query_filtered_parquet_from_azure,
    read_sample_parquet_data,
    create_interactive_data_query,
    download_query_data_csv,
    upload_csv_data,
    get_uploaded_data,
    get_assay_workflows_status,
)
from .analysis_plots import (
    parquet_seq_fov_heatmaps,
    create_workflow_selector,
    create_dataframe_selector,
    remove_outliers,
    remove_outliers_from_dataframe,
    create_outlier_filtering_controls,
    generate_chip_layout,
    map_fov_to_chip,
    plot_chip_heatmap_bokeh,
    chip_heatmap_controls,
    run_heatmap,
    plot_interactive_scatter,
    create_scatter_controls,
    create_comparison_mode_selector,
    plot_multi_timepoint_hist,
    symlog_bins,
    log_bins,
    create_multi_timepoint_hist_controls,
    plot_histogram_from_controls,
    create_time_series_controls,
    plot_feature_channels_over_time,
    plot_time_series_from_controls
)
from .groups_manager import (
    create_wells_groups_manager,
    filter_valid_groups,
    format_group_option
)
from .system_utils import (
    monitor_resources,
    get_memory_info,
    stop_if_memory_high
)

__version__ = "0.2.0"

__all__ = [
    "Sample",
    # Constants
    "CHANNEL_NAME_DICT",
    "CHANNEL_COLORS_DICT", 
    "PARQUET_FILES",
    "PARQUET_NON_FEATURE_COLS",
    "MAGNIFICATION_GRID_CONFIG",
    # API utilities
    "get_experiment_metadata", 
    "get_sample_metadata",
    "get_sample_datapath",
    "get_sample_records",
    "generate_token",
    "get_wells_groups",
    "save_wells_group",
    "remove_wells_group",
    "get_assay_workflows",
    # Metadata display
    "display_hardware_metadata",
    "display_sample_metadata",
    "display_serial_number_records",
    # Vizarr viewer
    "create_fovs_vizarr_viewer",
    # Data query
    "query_filtered_parquet_from_azure",
    "read_sample_parquet_data",
    "create_interactive_data_query",
    "download_query_data_csv",
    "get_assay_workflows_status",
    "upload_csv_data",
    "get_uploaded_data",
    # Analysis plots
    "parquet_seq_fov_heatmaps",
    "create_workflow_selector",
    "create_dataframe_selector",
    "remove_outliers",
    "remove_outliers_from_dataframe",
    "create_outlier_filtering_controls",
    "generate_chip_layout",
    "map_fov_to_chip",
    "plot_chip_heatmap_bokeh",
    "chip_heatmap_controls",
    "run_heatmap",
    "plot_interactive_scatter",
    "create_scatter_controls",
    "create_comparison_mode_selector",
    "plot_multi_timepoint_hist",
    "symlog_bins",
    "log_bins",
    "create_multi_timepoint_hist_controls",
    "plot_histogram_from_controls",
    "create_time_series_controls",
    "plot_feature_channels_over_time",
    "plot_time_series_from_controls",
    # Wells groups manager
    "create_wells_groups_manager",
    "filter_valid_groups",
    "format_group_option",
    # System utilities
    "monitor_resources",
    "get_memory_info",
    "stop_if_memory_high"
]