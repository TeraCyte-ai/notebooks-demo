"""
Sample class for TeraCyte data overview notebooks.

This module provides a Sample class that encapsulates all sample-related data,
metadata, and storage configuration for easy access across notebooks.
"""

import fsspec
import json
import pandas as pd
import pyarrow.parquet as pq
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from .constants import CHANNEL_NAME_DICT, PARQUET_FILES
from .api_utils import *

class Sample:
    """
    A class that encapsulates all sample-related data and metadata.
    
    This class provides a single interface to access sample tokens, metadata,
    storage paths, and Azure configuration for TeraCyte data analysis.
    
    Attributes:
        serial_number (str): The sample serial number
        exp_id (str): The experiment ID
        assay_id (str): The assay ID
        sample_token (dict): Token information for data access
        sample_datapath (str): Path to sample data
        account_name (str): Azure storage account name
        base_path (str): Base path in Azure storage
        sas (str): SAS token for Azure access
        exp_metadata (dict): Experiment metadata
        sample_metadata (dict): Sample metadata
        zarr_path (str): Path to the Zarr data store
    """
    
    def __init__(self, serial_number: str, exp_id: str, assay_id: str):
        """
        Initialize a Sample object with the given identifiers.
        
        Args:
            serial_number (str): The sample serial number
            exp_id (str): The experiment ID
            assay_id (str): The assay ID
        """
        self.serial_number = serial_number
        self.exp_id = exp_id
        self.assay_id = assay_id
        
        # Initialize all attributes
        self._sample_token: Optional[Dict[str, Any]] = None
        self._sample_datapath: Optional[str] = None
        self._account_name: Optional[str] = None
        self._base_path: Optional[str] = None
        self._sas: Optional[str] = None
        self._exp_metadata: Optional[Dict[str, Any]] = None
        self._sample_metadata: Optional[Dict[str, Any]] = None
        self._zarr_path: Optional[str] = None
        self._zarr_url: Optional[str] = None
        
        # Multiple parquet files support
        self._available_parquet_types: Optional[list] = None
        self._parquet_configs: Optional[Dict[str, Dict[str, Any]]] = None
        
        self._channels: Optional[Dict[int, str]] = None

        # Load data on initialization
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load all sample-related data and metadata."""
        # Generate token and datapath
        self._sample_token = generate_token(serial_number=self.serial_number)
        self._sample_datapath = get_sample_datapath(serial_number=self.serial_number)
        
        # Parse Azure configuration from token
        if self._sample_token:
            parsed_url = urlparse(self._sample_token['sas_url'])
            self._account_name = parsed_url.netloc.split('.')[0]
            self._base_path = self._sample_token['blob_path']
            self._sas = self._sample_token['sas_token']
        
        # Load metadata
        self._exp_metadata = get_experiment_metadata(exp_id=self.exp_id)
        self._sample_metadata = get_sample_metadata(
            serial_number=self.serial_number, 
            exp_id=self.exp_id
        )
        
        # Construct zarr and parquet paths and URLs
        if self._base_path and self._account_name and self._sas:
            self._zarr_path = f"{self._base_path}/{self.assay_id}/{self.assay_id}.zarr"
            self._zarr_url = f"https://{self._account_name}.blob.core.windows.net/{self._zarr_path}?{self._sas}"
            
            # Discover and load multiple parquet files
            self._discover_parquet_files()
            
            # Load metadata information
            self._load_channels()
    
    def _discover_parquet_files(self):
        """Discover and configure available parquet files for this sample."""
        try:
            if not self._base_path or not self._account_name or not self._sas:
                self._available_parquet_types = []
                self._parquet_configs = {}
                return
                
            fs = fsspec.filesystem("az", account_name=self._account_name, sas_token=self._sas)
            base_assay_path = f"{self._base_path}/{self.assay_id}"
            
            available_types = []
            configs = {}
            
            # Check for each known parquet file type
            for parquet_type, parquet_info in PARQUET_FILES.items():
                filename = parquet_info["filename"]
                parquet_path = f"{base_assay_path}/{filename}.parquet"
                
                try:
                    # Check if the parquet file/directory exists
                    if fs.exists(parquet_path):
                        available_types.append(parquet_type)
                        
                        # Configure this parquet type
                        configs[parquet_type] = {
                            "type": parquet_type,
                            "filename": filename,
                            "description": parquet_info["description"],
                            "path": parquet_path,
                            "url": f"https://{self._account_name}.blob.core.windows.net/{parquet_path}?{self._sas}",
                            "query_columns": parquet_info["query_columns"],
                            "metadata": None  # Will be loaded later if needed
                        }
                                                
                except Exception as e:
                    print(f"Error checking parquet file {filename}: {e}")
                    continue
                    
            self._available_parquet_types = available_types
            self._parquet_configs = configs
                            
        except Exception as e:
            print(f"Error discovering parquet files: {e}")
            self._available_parquet_types = []
            self._parquet_configs = {}
            
    def _load_parquet_metadata_for_type(self, parquet_type: str) -> Dict[str, Any]:
        """
        Load metadata for a specific parquet file type.
        
        Args:
            parquet_type (str): The type of parquet file (e.g., 'wells_data', 'cells_data', 'colocalization')
            
        Returns:
            dict: Metadata information for the specified parquet type
        """
        try:
            if parquet_type not in self._parquet_configs:
                return {"error": f"Parquet type '{parquet_type}' not available"}
                
            config = self._parquet_configs[parquet_type]
            parquet_path = config["path"]
            
            if not self._account_name or not self._sas:
                return {"error": "Missing Azure configuration"}
                
            fs = fsspec.filesystem("az", account_name=self._account_name, sas_token=self._sas)
            
            result = {
                "type": parquet_type,
                "filename": config["filename"],
                "description": config["description"],
                "columns": [],
                "partition_columns": [],
                "partition_values": {},
                "sequences": {},
                "assay_id": None,
                "query_columns": config["query_columns"]
            }

            # Try to read _metadata file first
            metadata_path = f"{parquet_path.rstrip('/')}/_metadata"
            if fs.exists(metadata_path):
                try:
                    with fs.open(metadata_path, 'rb') as f:
                        metadata_obj = pq.read_metadata(f)
                        schema = metadata_obj.schema.to_arrow_schema()
                        
                        # Get column names
                        result["columns"] = schema.names
                            
                except Exception as e:
                    print(f"Error reading _metadata for {parquet_type}: {e}")
            
            # Extract custom metadata
            custom_metadata_path = f"{parquet_path.rstrip('/')}/custom_metadata.json"
            if fs.exists(custom_metadata_path):
                try:
                    with fs.open(custom_metadata_path, 'rb') as f:
                        custom_metadata = json.load(f)
                        result["partition_columns"] = custom_metadata.get("partitions", [])
                        result["sequences"] = custom_metadata.get("sequences", {})
                        result["assay_id"] = custom_metadata.get("assay_id")
                except Exception as e:
                    print(f"Error reading custom_metadata for {parquet_type}: {e}")

            # If we have partition columns, scan the file structure to get actual partition values
            if result["partition_columns"]:
                try:
                    # Find all parquet files
                    parquet_files = [
                        path for path in fs.find(parquet_path)
                        if path.endswith(".parquet") and "/_" not in path
                    ]
                    
                    if parquet_files:
                        # Extract partition values from file paths
                        partition_values = {}
                        for file_path in parquet_files:
                            path_parts = file_path.split('/')
                            for part in path_parts:
                                if '=' in part:
                                    key, value = part.split('=', 1)
                                    if key in result["partition_columns"]:  # Only collect known partition columns
                                        if key not in partition_values:
                                            partition_values[key] = set()
                                        try:
                                            partition_values[key].add(int(value))
                                        except ValueError:
                                            partition_values[key].add(value)
                        
                        # Convert to sorted lists
                        result["partition_values"] = {
                            key: sorted(list(values)) 
                            for key, values in partition_values.items()
                        }
                    else:
                        print(f"No parquet files found in {parquet_path}")
                        
                except Exception as e:
                    print(f"Error scanning partition values for {parquet_type}: {e}")
            
            # Cache the metadata in the config
            self._parquet_configs[parquet_type]["metadata"] = result
            
            return result
            
        except Exception as e:
            return {"error": f"Could not load metadata for {parquet_type}: {str(e)}"}
    
    def _load_channels(self):
        """Load and cache channel display names from experiment metadata."""
        try:
            if not hasattr(self, '_exp_metadata') or not self._exp_metadata:
                self._channels = {}
                return
                
            channels_info = {}
            
            if 'channels' in self._exp_metadata:
                channels = self._exp_metadata['channels']
                
                if isinstance(channels, dict):
                    for key, channel_data in channels.items():
                        try:
                            channel_index = int(key)
                            if isinstance(channel_data, dict) and 'channelName' in channel_data:
                                channel_name = channel_data['channelName']
                                # Use constants mapping for display name
                                display_name = CHANNEL_NAME_DICT.get(channel_name, channel_name)
                                channels_info[channel_index] = display_name
                            else:
                                channels_info[channel_index] = str(channel_data)
                        except (ValueError, TypeError):
                            continue
                
                elif isinstance(channels, list):
                    for i, channel in enumerate(channels):
                        if isinstance(channel, dict) and 'channelName' in channel:
                            channel_name = channel['channelName']
                            # Use constants mapping for display name
                            display_name = CHANNEL_NAME_DICT.get(channel_name, channel_name)
                            channels_info[i] = display_name
                        elif isinstance(channel, dict) and 'name' in channel:
                            channel_name = channel['name']
                            # Use constants mapping for display name
                            display_name = CHANNEL_NAME_DICT.get(channel_name, channel_name)
                            channels_info[i] = display_name
                        else:
                            channels_info[i] = str(channel)
            
            self._channels = channels_info
            
        except Exception as e:
            print(f"Warning: Could not load channel names from metadata: {e}")
            self._channels = {}
    
    def get_partition_hierarchy(self, parquet_type: str = "wells_data") -> Dict[str, Any]:
        """
        Get the hierarchical structure of parquet partitions for a specific parquet type.
        
        This method analyzes the parquet file structure to understand the nested
        partition relationships. For example, which FOVs exist in each sequence.
        
        Args:
            parquet_type (str): The type of parquet file to analyze (default: "wells_data")
        
        Returns:
            dict: Hierarchical partition structure
            
        Example:
            >>> sample = Sample("serial", "exp", "assay")
            >>> hierarchy = sample.get_partition_hierarchy("wells_data")
            >>> # Result might look like:
            >>> # {
            >>> #   "sequence": {
            >>> #     0: {"fov": [0, 2]},
            >>> #     1: {"fov": [0, 5]}
            >>> #   }
            >>> # }
        """
        try:
            # Check if the parquet type is available
            if not self.has_parquet_type(parquet_type):
                return {"error": f"Parquet type '{parquet_type}' not available"}
            
            parquet_path = self.get_parquet_path(parquet_type)
            if not parquet_path or not self._account_name or not self._sas:
                return {"error": "Missing Azure configuration"}
            
            fs = fsspec.filesystem("az", account_name=self._account_name, sas_token=self._sas)
            
            # Find all parquet files
            parquet_files = [
                path for path in fs.find(parquet_path)
                if path.endswith(".parquet") and "/_" not in path
            ]
            
            if not parquet_files:
                return {"error": f"No parquet files found for {parquet_type}"}
            
            # Get partition columns from metadata
            metadata = self.get_parquet_metadata(parquet_type)
            partition_columns = metadata.get("partition_columns", [])
            
            if not partition_columns:
                return {"message": f"No partition columns found for {parquet_type}"}
            
            # Build hierarchy structure
            hierarchy = {}
            
            for file_path in parquet_files:
                # Extract partition values from file path
                path_parts = file_path.split('/')
                partition_values = {}
                
                for part in path_parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        if key in partition_columns:
                            try:
                                partition_values[key] = int(value)
                            except ValueError:
                                partition_values[key] = value
                
                # Build nested structure based on partition column order
                current_level = hierarchy
                
                for i, column in enumerate(partition_columns):
                    if column in partition_values:
                        value = partition_values[column]
                        
                        if i == 0:  # First level (e.g., sequence)
                            if column not in current_level:
                                current_level[column] = {}
                            if value not in current_level[column]:
                                current_level[column][value] = {}
                            current_level = current_level[column][value]
                        
                        else:  # Subsequent levels (e.g., fov)
                            if column not in current_level:
                                current_level[column] = set()
                            current_level[column].add(value)
            
            # Convert sets to sorted lists for consistent output
            def convert_sets_to_lists(obj):
                if isinstance(obj, dict):
                    return {k: convert_sets_to_lists(v) for k, v in obj.items()}
                elif isinstance(obj, set):
                    return sorted(list(obj))
                else:
                    return obj
            
            hierarchy = convert_sets_to_lists(hierarchy)
            
            return hierarchy
            
        except Exception as e:
            return {"error": f"Could not build partition hierarchy for {parquet_type}: {str(e)}"}
        
    @property
    def sample_token(self) -> Dict[str, Any]:
        """Get the sample token."""
        return self._sample_token or {}
    
    @property
    def sample_datapath(self) -> str:
        """Get the sample datapath."""
        return self._sample_datapath or ""
    
    @property
    def account_name(self) -> str:
        """Get the Azure storage account name."""
        return self._account_name or ""
    
    @property
    def base_path(self) -> str:
        """Get the base path in Azure storage."""
        return self._base_path or ""
    
    @property
    def sas(self) -> str:
        """Get the SAS token for Azure access."""
        return self._sas or ""
    
    @property
    def exp_metadata(self) -> Dict[str, Any]:
        """Get the experiment metadata."""
        return self._exp_metadata or {}
    
    @property
    def sample_metadata(self) -> Dict[str, Any]:
        """Get the sample metadata."""
        return self._sample_metadata or {}
    
    @property
    def zarr_path(self) -> str:
        """Get the path to the Zarr data store."""
        return self._zarr_path or ""
    
    @property
    def zarr_url(self) -> str:
        """Get the complete Azure blob URL for the Zarr data store."""
        return self._zarr_url or ""
    
    @property
    def channels(self) -> Dict[int, str]:
        """
        Get channel display names from experiment metadata.
        
        Returns:
            dict: Dictionary mapping channel index to display name
        """
        return self._channels or {}
    
    @property
    def available_parquet_types(self) -> list:
        """
        Get list of available parquet file types for this sample.
        
        Returns:
            list: List of parquet types (e.g., ['wells_data', 'cells_data', 'colocalization'])
        """
        return self._available_parquet_types or []
    
    @property
    def parquet_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all available parquet files.
        
        Returns:
            dict: Dictionary mapping parquet type to its configuration
        """
        return self._parquet_configs or {}
    
    def get_parquet_config(self, parquet_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific parquet file type.
        
        Args:
            parquet_type (str): The type of parquet file
            
        Returns:
            dict: Configuration for the specified parquet type, or empty dict if not found
        """
        return self._parquet_configs.get(parquet_type, {})
    
    def get_parquet_url(self, parquet_type: str) -> str:
        """
        Get the URL for a specific parquet file type.
        
        Args:
            parquet_type (str): The type of parquet file
            
        Returns:
            str: Azure blob URL with SAS token, or empty string if not found
        """
        config = self.get_parquet_config(parquet_type)
        return config.get("url", "")
    
    def get_parquet_path(self, parquet_type: str) -> str:
        """
        Get the path for a specific parquet file type.
        
        Args:
            parquet_type (str): The type of parquet file
            
        Returns:
            str: Path to the parquet file, or empty string if not found
        """
        config = self.get_parquet_config(parquet_type)
        return config.get("path", "")
    
    def get_parquet_metadata(self, parquet_type: str) -> Dict[str, Any]:
        """
        Get metadata for a specific parquet file type.
        
        Args:
            parquet_type (str): The type of parquet file
            
        Returns:
            dict: Metadata for the specified parquet type
        """
        config = self.get_parquet_config(parquet_type)
        if config and config.get("metadata") is None:
            # Load metadata if not already loaded
            self._load_parquet_metadata_for_type(parquet_type)
            config = self.get_parquet_config(parquet_type)  # Refresh config
        
        return config.get("metadata", {})
    
    def get_parquet_query_columns(self, parquet_type: str) -> list:
        """
        Get query columns for a specific parquet file type.
        
        Args:
            parquet_type (str): The type of parquet file
            
        Returns:
            list: List of column names for querying
        """
        config = self.get_parquet_config(parquet_type)
        return config.get("query_columns", [])
    
    def has_parquet_type(self, parquet_type: str) -> bool:
        """
        Check if a specific parquet file type is available.
        
        Args:
            parquet_type (str): The type of parquet file to check
            
        Returns:
            bool: True if the parquet type is available, False otherwise
        """
        return parquet_type in self.available_parquet_types
    
    def get_parquet_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all available parquet files.
        
        Returns:
            dict: Summary information about available parquet files with each parquet type as a key
        """
        from .constants import PARQUET_FILES
        
        summary = {}
        
        # Check all possible parquet types from constants
        for parquet_type, config in PARQUET_FILES.items():
            is_available = parquet_type in self.available_parquet_types
            
            summary[parquet_type] = {
                "available": is_available,
                "filename": config.get("filename", ""),
                "description": config.get("description", ""),
            }
            
            # Add metadata if available
            if is_available:
                try:
                    metadata = self.get_parquet_metadata(parquet_type)
                    if metadata and "error" not in metadata:
                        summary[parquet_type]["metadata"] = metadata
                except Exception:
                    # Metadata loading failed, but file is still available
                    pass
        
        return summary

    def _initialize_dataframes_dict(self):
        """Initialize the dataframes dictionary with empty DataFrames for all available parquet types."""
        if not hasattr(self, '_dataframes'):
            import pandas as pd
            from .constants import PARQUET_FILES
            
            self._dataframes = {}
            self._dataframe_filters = {}
            
            # Initialize empty DataFrames for all possible parquet types
            for parquet_type in PARQUET_FILES.keys():
                self._dataframes[parquet_type] = pd.DataFrame()  # Empty DataFrame
                self._dataframe_filters[parquet_type] = {}
    
    def set_dataframe(self, parquet_type: str, dataframe, filters: Dict[str, Any] = None):
        """
        Store a DataFrame for a specific parquet type.
        
        Args:
            parquet_type (str): The parquet type ('wells_data', 'cells_data', 'colocalization')
            dataframe: The pandas DataFrame to store
            filters: The filters used to generate this data (optional)
        """
        self._initialize_dataframes_dict()
        self._dataframes[parquet_type] = dataframe
        self._dataframe_filters[parquet_type] = filters or {}

    def get_dataframe(self, parquet_type: str):
        """
        Get the DataFrame for a specific parquet type.
        
        Args:
            parquet_type (str): The parquet type to retrieve
            
        Returns:
            pd.DataFrame: The stored DataFrame (empty if not loaded)
        """
        self._initialize_dataframes_dict()
        return self._dataframes.get(parquet_type, pd.DataFrame())
    
    def get_dataframe_filters(self, parquet_type: str) -> Dict[str, Any]:
        """
        Get the filters used for a specific parquet type DataFrame.
        
        Args:
            parquet_type (str): The parquet type
            
        Returns:
            dict: The filters used for this DataFrame
        """
        self._initialize_dataframes_dict()
        return self._dataframe_filters.get(parquet_type, {})
    
    def is_dataframe_loaded(self, parquet_type: str) -> bool:
        """
        Check if a DataFrame is loaded (non-empty) for a specific parquet type.
        
        Args:
            parquet_type (str): The parquet type to check
            
        Returns:
            bool: True if DataFrame is loaded and non-empty, False otherwise
        """
        self._initialize_dataframes_dict()
        df = self._dataframes.get(parquet_type)
        return df is not None and not df.empty
    
    def clear_dataframe(self, parquet_type: str):
        """
        Clear (reset to empty) the DataFrame for a specific parquet type.
        
        Args:
            parquet_type (str): The parquet type to clear
        """
        import pandas as pd
        self._initialize_dataframes_dict()
        self._dataframes[parquet_type] = pd.DataFrame()
        self._dataframe_filters[parquet_type] = {}
    
    def clear_all_dataframes(self):
        """Clear all loaded DataFrames."""
        self._initialize_dataframes_dict()
        import pandas as pd
        
        for parquet_type in self._dataframes.keys():
            self._dataframes[parquet_type] = pd.DataFrame()
            self._dataframe_filters[parquet_type] = {}
        
    def get_dataframes_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all loaded DataFrames.
        
        Returns:
            dict: Summary information for each parquet type
        """
        self._initialize_dataframes_dict()
        summary = {}
        
        for parquet_type, df in self._dataframes.items():
            is_available = parquet_type in self.available_parquet_types
            is_loaded = not df.empty
            
            summary[parquet_type] = {
                'available': is_available,
                'loaded': is_loaded,
                'shape': df.shape if is_loaded else (0, 0),
                'columns': list(df.columns) if is_loaded else [],
                'filters': self._dataframe_filters.get(parquet_type, {})
            }
        
        return summary
    
    @property
    def all_dataframes(self) -> Dict[str, Any]:
        """Get all loaded DataFrames as a dictionary."""
        self._initialize_dataframes_dict()
        return self._dataframes.copy()
    
    def get_fov_zarr_path(self, row: str, col: str, field: str = "0") -> str:
        """
        Get the full Zarr path for a specific FOV.
        
        Args:
            row (str): Row identifier (e.g., 'A', 'B', 'C')
            col (str): Column identifier (e.g., '1', '2', '3')
            field (str): Field identifier (default: '0')
        
        Returns:
            str: Full path to the FOV Zarr data
        """
        return f"{self.zarr_path}/{row}/{col}/{field}"
    
    def get_fov_source_url(self, row: str, col: str, field: str = "0") -> str:
        """
        Get the complete Azure blob URL for a specific FOV.
        
        Args:
            row (str): Row identifier (e.g., 'A', 'B', 'C')
            col (str): Column identifier (e.g., '1', '2', '3')
            field (str): Field identifier (default: '0')
        
        Returns:
            str: Complete Azure blob URL with SAS token
        """
        fov_path = self.get_fov_zarr_path(row, col, field)
        return f"https://{self.account_name}.blob.core.windows.net/{fov_path}?{self.sas}"
    
    def get_zarr_source_url(self) -> str:
        """
        Get the complete Azure blob URL for the entire Zarr store.
        
        Returns:
            str: Complete Azure blob URL with SAS token
        """
        return self.zarr_url
    
    def get_storage_options(self) -> Dict[str, str]:
        """
        Get the storage options dictionary for fsspec.
        
        Returns:
            dict: Storage options with account name and SAS token
        """
        return {
            'account_name': self.account_name,
            'sas_token': self.sas
        }
    
    @property
    def wells_groups(self):
        """
        Get all wells groups for this sample.
        
        Returns:
            list: List of wells groups
            
        Example:
            >>> sample = Sample("serial", "exp", "assay")
            >>> groups = sample.wells_groups
            >>> for group in groups:
            ...     print(f"Group: {group['query_name']}")
        """
        return get_wells_groups(
            serial_number=self.serial_number,
            assay_id=self.assay_id
        )
    
    def create_wells_group(self, name: str, description: str, global_indexes: list):
        """
        Create a new wells group for this sample.
        
        Args:
            name (str): Name for the new group
            description (str): Description for the new group
            global_indexes (list): List of global well indexes
            
        Returns:
            dict: API response
            
        Example:
            >>> sample = Sample("serial", "exp", "assay")
            >>> result = sample.create_wells_group(
            ...     name="my_group",
            ...     description="My analysis group",
            ...     global_indexes=[100, 200, 300]
            ... )
        """
        return save_wells_group(
            serial_number=self.serial_number,
            assay_id=self.assay_id,
            group_name=name,
            group_description=description,
            global_indexes=global_indexes
        )
    
    def delete_wells_group(self, name: str):
        """
        Delete a wells group for this sample.
        
        Args:
            name (str): Name of the group to delete
            
        Returns:
            dict: API response
            
        Example:
            >>> sample = Sample("serial", "exp", "assay")
            >>> result = sample.delete_wells_group("my_group")
        """
        return remove_wells_group(
            serial_number=self.serial_number,
            assay_id=self.assay_id,
            group_name=name
        )
    
    def get_wells_group_by_name(self, name: str):
        """
        Get a specific wells group by name.
        
        Args:
            name (str): Name of the group to find
            
        Returns:
            dict or None: Group data if found, None otherwise
            
        Example:
            >>> sample = Sample("serial", "exp", "assay")
            >>> group = sample.get_wells_group_by_name("my_group")
        """
        groups = self.wells_groups
        if groups:
            for group in groups:
                if group.get("query_name") == name:
                    return group
        return None
    
    def list_wells_group_names(self):
        """
        Get a list of all wells group names.
        
        Returns:
            list: List of group names
            
        Example:
            >>> sample = Sample("serial", "exp", "assay")
            >>> names = sample.list_wells_group_names()
        """
        groups = self.wells_groups
        if groups:
            return [group.get("query_name", "") for group in groups]
        return []
    
    def get_wells_groups_summary(self):
        """
        Get a summary of all wells groups.
        
        Returns:
            dict: Summary information about groups
            
        Example:
            >>> sample = Sample("serial", "exp", "assay")
            >>> summary = sample.get_wells_groups_summary()
            >>> print(f"Total groups: {summary['total_groups']}")
        """
        groups = self.wells_groups
        if not groups:
            return {
                "total_groups": 0,
                "groups": []
            }
        
        summary = {
            "total_groups": len(groups),
            "groups": []
        }
        
        for group in groups:
            group_info = {
                "name": group.get("query_name", ""),
                "description": group.get("query_description", ""),
                "wells_count": len(group.get("global_indexes", [])),
                "query_index": group.get("query_index", None)
            }
            summary["groups"].append(group_info)
        
        return summary
    
    def refresh(self):
        """Refresh all sample data by reloading from the API."""
        self._load_sample_data()
    
    def __repr__(self) -> str:
        """String representation of the Sample object."""
        return (f"Sample(serial_number='{self.serial_number}', "
                f"exp_id='{self.exp_id}', assay_id='{self.assay_id}')")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"TeraCyte Sample\n"
                f"  Serial Number: {self.serial_number}\n"
                f"  Experiment ID: {self.exp_id}\n"
                f"  Assay ID: {self.assay_id}\n")
