"""Custom exceptions for Connectomix."""


class ConnectomixError(Exception):
    """Base exception for Connectomix."""
    pass


class BIDSError(ConnectomixError):
    """Error related to BIDS dataset."""
    pass


class ConfigurationError(ConnectomixError):
    """Error in configuration."""
    pass


class PreprocessingError(ConnectomixError):
    """Error during preprocessing."""
    pass


class ConnectivityError(ConnectomixError):
    """Error during connectivity analysis."""
    pass


class StatisticalError(ConnectomixError):
    """Error during statistical analysis."""
    pass
