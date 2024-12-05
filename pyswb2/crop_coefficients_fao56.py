
import numpy as np
import pandas as pd

class CropCoefficientsFAO56:
    """Class to handle crop coefficients and reference ET modifications based on FAO-56."""

    def __init__(self):
        """Initialize crop coefficients and simulation parameters."""
        self.crop_coefficients = {}  # Dictionary to hold coefficients for each crop type
        self.et_reference = None     # Reference evapotranspiration values

    def set_reference_et(self, et_values):
        """Set reference evapotranspiration values.

        Args:
            et_values (array-like): Reference ET values.
        """
        self.et_reference = np.array(et_values, dtype=np.float32)

    def add_crop_coefficient(self, crop_type, coefficient):
        """Add or update a crop coefficient.

        Args:
            crop_type (str): Name of the crop.
            coefficient (float): Crop coefficient value.
        """
        self.crop_coefficients[crop_type] = coefficient

    def update_et_for_crop(self, crop_type):
        """Update ET values using the crop coefficient for a specific crop.

        Args:
            crop_type (str): Name of the crop.

        Returns:
            np.ndarray: Adjusted ET values for the crop.
        """
        if self.et_reference is None:
            raise ValueError("Reference ET values must be set before updating.")
        
        if crop_type not in self.crop_coefficients:
            raise KeyError(f"Crop type '{crop_type}' not found in coefficients.")
        
        coefficient = self.crop_coefficients[crop_type]
        return self.et_reference * coefficient

    def get_coefficient(self, crop_type):
        """Get the coefficient for a specific crop.

        Args:
            crop_type (str): Name of the crop.

        Returns:
            float: Crop coefficient.
        """
        return self.crop_coefficients.get(crop_type, None)
