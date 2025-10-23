"""
Configuration management for TeraCyte Notebooks Utils package.
Simple configuration system that can be extended later with secure solutions.
"""
import os
from typing import Optional


class Config:
    """Configuration class to manage API settings."""
    
    def __init__(self):
        self._service_ip = None
        # Default can be set to None - users must configure
        self._default_ip = None
        
    @property
    def service_ip(self) -> str:
        """
        Get the service IP address.
        
        Priority order:
        1. Previously set value via set_service_ip()
        2. Environment variable TERACYTE_SERVICE_IP
        3. Default IP (if set)
        4. Raises ValueError if not configured
        
        Returns:
            str: The service IP address
            
        Raises:
            ValueError: If service IP is not configured
        """
        # First check if manually set
        if self._service_ip:
            return self._service_ip
            
        # Check environment variable
        env_ip = os.getenv('TERACYTE_SERVICE_IP')
        if env_ip:
            return env_ip
            
        # Check default (can be None for now)
        if self._default_ip:
            return self._default_ip
            
        raise ValueError(
            "TeraCyte service IP not configured. Please set it using one of these methods:\n"
            "1. Set environment variable: export TERACYTE_SERVICE_IP='your.ip.address'\n"
            "2. Use teracyte_notebooks_utils.set_service_ip('your.ip.address') in your code\n"
            "3. Contact TeraCyte support for configuration assistance"
        )
    
    def set_service_ip(self, ip_address: str) -> None:
        """
        Set the service IP address programmatically.
        
        Args:
            ip_address (str): The IP address or hostname of the TeraCyte service
        """
        if not ip_address or not isinstance(ip_address, str):
            raise ValueError("IP address must be a non-empty string")
        self._service_ip = ip_address
    
    def set_default_ip(self, ip_address: str) -> None:
        """
        Set the default IP address (for internal use).
        
        Args:
            ip_address (str): The default IP address
        """
        self._default_ip = ip_address
    
    def is_configured(self) -> bool:
        """
        Check if the service IP is configured.
        
        Returns:
            bool: True if service IP is configured, False otherwise
        """
        try:
            _ = self.service_ip
            return True
        except ValueError:
            return False


# Global configuration instance
config = Config()


def get_service_ip() -> str:
    """
    Get the configured service IP address.
    
    Returns:
        str: The service IP address
        
    Raises:
        ValueError: If service IP is not configured
    """
    return config.service_ip


def set_service_ip(ip_address: str) -> None:
    """
    Set the service IP address.
    
    Args:
        ip_address (str): The IP address or hostname of the TeraCyte service
    """
    config.set_service_ip(ip_address)


def is_configured() -> bool:
    """
    Check if the service is configured.
    
    Returns:
        bool: True if configured, False otherwise
    """
    return config.is_configured()