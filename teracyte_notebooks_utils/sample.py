"""
Sample class for TeraCyte data overview notebooks.

This module provides a Sample class that encapsulates all sample-related data,
metadata, and storage configuration for easy access across notebooks.
"""

import fsspec
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re
from collections import defaultdict
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from .constants import CHANNEL_NAME_DICT, PARQUET_FILES, PARQUET_NON_FEATURE_COLS
from .api_utils import *

class Sample:
    """
    A class that encapsulates all sample-related data and metadata.
    
    This class provides a single interface to access sample metadata,
    storage paths, and Azure configuration for TeraCyte data analysis.
    
    Attributes:
        serial_number (str): The sample serial number
        exp_id (str): The experiment ID
        assay_id (str): The assay ID
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
    
    def _get_parquet_metadata(self, fs, parquet_path):
        """
        Get comprehensive parquet metadata including columns, rows, partitions, and partition values.
        Efficiently extracts partition values from PyArrow metadata file paths.
        
        Args:
            fs: The filesystem object
            parquet_path (str): Path to the parquet dataset
            
        Returns:
            dict: Dictionary containing columns, num_rows, partition_columns, partition_values, and custom_metadata
        """
        try:
            meta_path = f"{parquet_path}/_metadata"
            common_path = f"{parquet_path}/_common_metadata"
            
            # Check if _metadata file exists
            if not fs.exists(meta_path):
                return {
                    "columns": [],
                    "num_rows": 0,
                    "partition_columns": [],
                    "partition_values": {},
                    "custom_metadata": {}
                }
            
            # Create PyArrow filesystem wrapper and load metadata (single load)
            pa_fs = pa.fs.PyFileSystem(pa.fs.FSSpecHandler(fs))
            pf_meta = pq.ParquetFile(meta_path, filesystem=pa_fs)
            md = pf_meta.metadata
            
            # Number of rows
            num_rows = sum(md.row_group(i).num_rows for i in range(md.num_row_groups))
            
            # Column list (from _common_metadata if available, otherwise from _metadata)
            try:
                pf_common = pq.ParquetFile(common_path, filesystem=pa_fs)
                schema = pf_common.schema
            except Exception:
                schema = pf_meta.schema
            all_columns = schema.names
            
            # Load custom metadata
            custom_metadata_path = f"{parquet_path.rstrip('/')}/custom_metadata.json"
            if fs.exists(custom_metadata_path):
                try:
                    with fs.open(custom_metadata_path, 'rb') as f:
                        custom_metadata = json.load(f)
                except Exception:
                    custom_metadata = {}
            else:
                custom_metadata = {}
            
            # Get expected partitions from custom metadata
            expected_partitions = custom_metadata.get("partition_columns", None)
            
            # Extract partition values directly from PyArrow metadata file paths            
            partition_values = defaultdict(set)
            partition_cols = set()
            
            # Extract file paths from all row groups (using already loaded metadata)
            paths = []
            for i in range(md.num_row_groups):
                rg = md.row_group(i)
                col0 = rg.column(0)  # Any column works for file_path
                if hasattr(col0, 'file_path') and col0.file_path:
                    paths.append(col0.file_path)
            
            # Parse partition values from paths
            for path in set(paths):  # Use set to avoid duplicates
                for segment in path.split('/'):
                    if '=' in segment:
                        key, value = segment.split('=', 1)
                        # Accept partition if it's in expected list or if we have no expected list
                        if expected_partitions is None or key in expected_partitions:
                            partition_cols.add(key)
                            partition_values[key].add(value)
            
            # Convert to sorted lists, prioritizing expected partition order if available
            if expected_partitions:
                expected_found = [col for col in expected_partitions if col in partition_cols]
                other_found = [col for col in sorted(partition_cols) if col not in expected_partitions]
                partition_columns = expected_found + other_found
            else:
                partition_columns = sorted(partition_cols)
            
            partition_values_dict = {k: sorted(v, key=lambda x: int(x) if x.isdigit() else x) for k, v in partition_values.items()}
            
            for col in partition_columns:
                values = partition_values_dict[col]
                sample_values = values[:3] + ['...'] if len(values) > 3 else values
            
            # Combine data columns and partition columns for complete column list
            all_available_columns = list(all_columns) + partition_columns
            
            return {
                "columns": all_available_columns,
                "num_rows": num_rows,
                "partition_columns": partition_columns,
                "partition_values": partition_values_dict,
                "custom_metadata": custom_metadata
            }
            
        except Exception as e:
            print(f"Error getting parquet metadata: {e}")
            return {
                "columns": [],
                "num_rows": 0,
                "partition_columns": [],
                "partition_values": {},
                "custom_metadata": {}
            }
    




    def _get_complete_parquet_metadata(self, fs, parquet_path, parquet_type, parquet_info):
        """
        Get complete parquet metadata combining _metadata file info and directory scanning.
        
        Args:
            fs: The filesystem object
            parquet_path (str): Path to the parquet dataset
            parquet_type (str): The type of parquet file
            parquet_info (dict): Info from PARQUET_FILES constants
            
        Returns:
            dict: Complete metadata structure with all parquet and custom information
        """
        try:
            # Start with the basic structure
            result = {
                "type": parquet_type,
                "filename": parquet_info["filename"],
                "description": parquet_info["description"],
                "columns": [],
                "partition_columns": [],
                "partition_values": {},
                "sequences": {},
                "feature_columns": []
            }
            
            # Get basic parquet metadata (includes custom metadata, extracts partitions from PyArrow metadata)
            basic_metadata = self._get_parquet_metadata(fs, parquet_path)
            result["columns"] = basic_metadata["columns"]
            result["partition_columns"] = basic_metadata["partition_columns"]  
            result["partition_values"] = basic_metadata["partition_values"]
            result["num_rows"] = basic_metadata["num_rows"]
            
            # Calculate feature columns (all columns except non-feature columns)
            all_available_columns = basic_metadata["columns"]
            feature_columns = [col for col in all_available_columns if col not in PARQUET_NON_FEATURE_COLS]
            result["feature_columns"] = feature_columns
            
            # Extract sequences and other info from custom metadata (already loaded)
            custom_metadata = basic_metadata.get("custom_metadata", {})
            result["sequences"] = custom_metadata.get("sequences", {})
            
            return result
            
        except Exception as e:
            print(f"Error getting complete parquet metadata for {parquet_type}: {e}")
            # Return minimal structure on error
            return {
                "type": parquet_type,
                "filename": parquet_info["filename"],
                "description": parquet_info["description"],
                "columns": [],
                "partition_columns": [],
                "partition_values": {},
                "sequences": {},
                "feature_columns": []
            }

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
                        
                        # Get complete metadata (includes both basic parquet info and directory scanning)
                        complete_metadata = self._get_complete_parquet_metadata(fs, parquet_path, parquet_type, parquet_info)
                        
                        # Configure this parquet type
                        configs[parquet_type] = {
                            "type": parquet_type,
                            "filename": filename,
                            "description": parquet_info["description"],
                            "path": parquet_path,
                            "url": f"https://{self._account_name}.blob.core.windows.net/{parquet_path}?{self._sas}",
                            "feature_columns": complete_metadata.get("feature_columns", []),
                            "columns": complete_metadata["columns"],
                            "num_rows": complete_metadata.get("num_rows", 0),
                            "partition_columns": complete_metadata["partition_columns"],
                            "partition_values": complete_metadata["partition_values"],
                            "sequences": complete_metadata.get("sequences", {})
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
        Get metadata for all available parquet files.
        
        Returns:
            dict: Dictionary mapping parquet type to its metadata
        """
        return self._parquet_configs or {}
    
    def get_parquet_metadata(self, parquet_type: str) -> Dict[str, Any]:
        """
        Get metadata for a specific parquet file type.
        
        Args:
            parquet_type (str): The type of parquet file
            
        Returns:
            dict: Metadata for the specified parquet type, or empty dict if not found
        """
        return self._parquet_configs.get(parquet_type, {})
    
    def get_parquet_config(self, parquet_type: str) -> Dict[str, Any]:
        """
        DEPRECATED: Use get_parquet_metadata() instead.
        
        Args:
            parquet_type (str): The type of parquet file
            
        Returns:
            dict: Metadata for the specified parquet type, or empty dict if not found
        """
        return self.get_parquet_metadata(parquet_type)
    
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
                    if metadata:
                        # Include key metadata fields in summary
                        summary[parquet_type]["columns"] = metadata.get("columns", [])
                        summary[parquet_type]["num_rows"] = metadata.get("num_rows", 0)
                        summary[parquet_type]["partition_columns"] = metadata.get("partition_columns", [])
                        summary[parquet_type]["sequences"] = metadata.get("sequences", {})
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
