CHANNEL_NAME_DICT = {"50R50T Beamsplitter non polarizing 400-750": "BF",
                     "mCherryTxRed Filter Set": "mCherry",
                     "GFP Filter Set": "GFP",
                     "CFP": "CFP",
                     "YFP ET Filter Set": "YFP",
                     "DAPI Filter Set": "DAPI",
                     "Cy5 Filter Sets": "Cy5",
                     "Cy7": "Cy7"
                     }

CHANNEL_COLORS_DICT = {"50R50T Beamsplitter non polarizing 400-750": "FFFFFF",  # white reference
                       "mCherryTxRed Filter Set": "FF4D4D",                     # bright red
                       "GFP Filter Set": "00FF80",                              # green
                       "CFP": "00BFFF",                                         # deep sky blue
                       "YFP ET Filter Set": "FFFF66",                           # bright yellow
                       "DAPI Filter Set": "3366FF",                             # medium blue
                       "Cy5 Filter Sets": "9900CC",                             # purple-magenta
                       "Cy7": "660066"                                          # deep purple
                       }

# Supported parquet files and their configurations
PARQUET_FILES = {
    "wells_data": {
        "filename": "wells_data",
        "description": "Well-level analysis data",
        "query_columns": [
            'well_area', 'well_intensity_mean', 'well_intensity_min', 'well_intensity_max', 'well_intensity_median',
            'local_intensity_mean_background', 'local_intensity_min_background', 'local_intensity_max_background',
            'local_intensity_median_background', 'intensity_global_mean_background', 'intensity_global_median_background',
            'global_CTCF_mean', 'global_CTCF_median', 'local_CTCF_mean', 'local_CTCF_median', 'empty_well_CTCF',
            'moment_variance', 'raw_moment_variance', 'intensity_weighted_distance',
            'x', 'y', 'r', 'local_index', 'block_index', 'assay_index',
            'zplane', 'sequence', 'fov', 'channel_index', 'z_index', 'global_index', 'classification', 'probability'
        ],
        "feature_columns": [
            'well_area', 'well_intensity_mean', 'well_intensity_min', 'well_intensity_max', 'well_intensity_median',
            'local_intensity_mean_background', 'local_intensity_min_background', 'local_intensity_max_background',
            'local_intensity_median_background', 'intensity_global_mean_background', 'intensity_global_median_background',
            'global_CTCF_mean', 'global_CTCF_median', 'local_CTCF_mean', 'local_CTCF_median', 'empty_well_CTCF',
            'moment_variance', 'raw_moment_variance', 'intensity_weighted_distance'
        ]
    },
    "cells_data": {
        "filename": "cells_data",
        "description": "Cell-level analysis data",
        "query_columns": [
            'x', 'y', 'r', 'local_index', 'block_index',
            'zplane', 'img_correction', 'mask_type',
            'y_min', 'x_min', 'y_max', 'x_max', 'perimeter',
            'circularity', 'min_radius', 'max_radius', 'fov_object_center',
            'well_object_center', 'object_area', 'object_intensity_mean',
            'object_intensity_min', 'object_intensity_max',
            'object_intensity_median', 'object_intensity_stdev',
            'object_saturation_rate', 'object_entropy', 'area_inner_background',
            'intensity_mean_inner_background', 'intensity_min_inner_background',
            'intensity_max_inner_background', 'intensity_median_inner_background',
            'intensity_stdev_inner_background', 'inner_CTCF_mean',
            'inner_CTCF_median', 'inner_CTCF_stdev', 'object_moment_variance',
            'object_raw_moment_variance', 'inner_index', 'classification', 'conf',
            'sequence_timestamp', 'sequence', 'fov', 'channel_index', 'z_index', 'global_index', 'object_id'],
        "feature_columns": [
            'perimeter', 'circularity', 'min_radius', 'max_radius', 'fov_object_center',
            'well_object_center', 'object_area', 'object_intensity_mean',
            'object_intensity_min', 'object_intensity_max',
            'object_intensity_median', 'object_intensity_stdev',
            'object_saturation_rate', 'object_entropy', 'area_inner_background',
            'intensity_mean_inner_background', 'intensity_min_inner_background',
            'intensity_max_inner_background', 'intensity_median_inner_background',
            'intensity_stdev_inner_background', 'inner_CTCF_mean',
            'inner_CTCF_median', 'inner_CTCF_stdev', 'object_moment_variance',
            'object_raw_moment_variance']
    },
    "colocalization": {
        "filename": "colocalization",
        "description": "Colocalization analysis data",
        "query_columns": [
            "M1", "M2", "containment1", "containment2",
            "x", "y", "r", "zplane",
            "local_index", "block_index",
            "sequence", "fov", "label1", "label2", "threshold1", "threshold2",
            "channel_index1", "channel_index2", "z_index", "global_index"
        ],
        "feature_columns": ["M1", "M2", "containment1", "containment2"]
    }
}

# Default parquet file types to check for availability
DEFAULT_PARQUET_TYPES = ["wells_data", "cells_data", "colocalization"]

# Mapping of magnification to grid dimensions
MAGNIFICATION_GRID_CONFIG = {
    4: {"cols": 5, "rows": 6, "fov_count": 30},
    10: {"cols": 11, "rows": 14, "fov_count": 154},
}
