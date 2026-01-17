"""Configuration parameter validation."""

from typing import Any, List
from pathlib import Path


class ConfigValidator:
    """Validate configuration parameters.
    
    Accumulates validation errors and can raise them all at once.
    
    Attributes:
        errors: List of validation error messages
    """
    
    def __init__(self):
        """Initialize validator with empty error list."""
        self.errors: List[str] = []
    
    def validate_alpha(self, value: float, name: str) -> bool:
        """Validate alpha value is in [0, 1].
        
        Args:
            value: Value to validate
            name: Parameter name for error message
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, (int, float)):
            self.errors.append(f"{name} must be a number, got {type(value).__name__}")
            return False
        
        if not 0 <= value <= 1:
            self.errors.append(f"{name} must be between 0 and 1, got {value}")
            return False
        
        return True
    
    def validate_positive(self, value: float, name: str) -> bool:
        """Validate value is positive.
        
        Args:
            value: Value to validate
            name: Parameter name for error message
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, (int, float)):
            self.errors.append(f"{name} must be a number, got {type(value).__name__}")
            return False
        
        if value <= 0:
            self.errors.append(f"{name} must be positive, got {value}")
            return False
        
        return True
    
    def validate_non_negative(self, value: float, name: str) -> bool:
        """Validate value is non-negative.
        
        Args:
            value: Value to validate
            name: Parameter name for error message
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, (int, float)):
            self.errors.append(f"{name} must be a number, got {type(value).__name__}")
            return False
        
        if value < 0:
            self.errors.append(f"{name} must be non-negative, got {value}")
            return False
        
        return True
    
    def validate_file_exists(self, path: Path, name: str) -> bool:
        """Validate file exists.
        
        Args:
            path: Path to validate
            name: Parameter name for error message
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(path, Path):
            path = Path(path)
        
        if not path.exists():
            self.errors.append(f"{name} file not found: {path}")
            return False
        
        if not path.is_file():
            self.errors.append(f"{name} is not a file: {path}")
            return False
        
        return True
    
    def validate_dir_exists(self, path: Path, name: str) -> bool:
        """Validate directory exists.
        
        Args:
            path: Path to validate
            name: Parameter name for error message
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(path, Path):
            path = Path(path)
        
        if not path.exists():
            self.errors.append(f"{name} directory not found: {path}")
            return False
        
        if not path.is_dir():
            self.errors.append(f"{name} is not a directory: {path}")
            return False
        
        return True
    
    def validate_choice(self, value: Any, choices: List[Any], name: str) -> bool:
        """Validate value is in allowed choices.
        
        Args:
            value: Value to validate
            choices: List of allowed values
            name: Parameter name for error message
        
        Returns:
            True if valid, False otherwise
        """
        if value not in choices:
            self.errors.append(
                f"{name} must be one of {choices}, got '{value}'"
            )
            return False
        
        return True
    
    def validate_type(self, value: Any, expected_type: type, name: str) -> bool:
        """Validate value is of expected type.
        
        Args:
            value: Value to validate
            expected_type: Expected type
            name: Parameter name for error message
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, expected_type):
            self.errors.append(
                f"{name} must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            return False
        
        return True
    
    def raise_if_errors(self) -> None:
        """Raise ValueError if any validation errors occurred.
        
        Raises:
            ValueError: If there are any validation errors
        """
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in self.errors
            )
            raise ValueError(error_msg)
