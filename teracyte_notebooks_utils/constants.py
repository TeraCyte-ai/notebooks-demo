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
        "description": "Well-level analysis data"
    },
    "cells_data": {
        "filename": "cells_data",
        "description": "Cell-level analysis data",
    },
    "colocalization": {
        "filename": "colocalization",
        "description": "Colocalization analysis data",
    }
}

PARQUET_NON_FEATURE_COLS = [
    # IDs
    'exp_id', 'seq_id', 'fov_id', 'assay_id', 'image_id', 'mask_id', 'object_id',
    # Indexes
    'local_index', 'block_index', 'assay_index', 'global_index', 'inner_index', 'z_index', 'channel_index',
    # Coordinates and spatial
    'x', 'y', 'r', 'well_radius', 'zplane', 'fov',
    # Classification and probability
    'classification_model', 'probability', 'occupancy_probability', 'occupancy_classification', 'mask_type', 'classification',
    # Sequence and time
    'sequence', 'timepoint', 'timepoint_timestamp',
    # Image processing
    'img_correction', 'type'
]

# Mapping of magnification to grid dimensions
MAGNIFICATION_GRID_CONFIG = {
    4: {"cols": 5, "rows": 6, "fov_count": 30},
    10: {"cols": 11, "rows": 14, "fov_count": 154},
}

# Timepoint names
TIME_COL_NAME = ["timepoint", "sequence"]
