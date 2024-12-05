# Rooting Depth Module - FAO56
# This script provides support for dynamic rooting depth calculation

class RootingDepthFAO56:
    def __init__(self, crop_params, soil_params):
        """
        Initialize rooting depth with crop and soil parameters.

        :param crop_params: Dictionary of crop-specific parameters
        :param soil_params: Dictionary of soil-specific parameters
        """
        self.crop_params = crop_params
        self.soil_params = soil_params
        self.current_depth = 0

    def initialize_rooting_depth(self, initial_depth):
        """
        Initialize the rooting depth.

        :param initial_depth: Initial depth of the root zone (e.g., in mm)
        """
        self.current_depth = initial_depth
        print(f"Rooting depth initialized to {initial_depth} mm.")

    def update_rooting_depth(self, growth_stage, growth_rate):
        """
        Update the rooting depth based on plant growth stage and growth rate.

        :param growth_stage: Current growth stage of the plant
        :param growth_rate: Rate of root depth growth (mm/day)
        """
        # Example calculation (can be expanded based on crop/soil properties)
        self.current_depth += growth_rate
        print(f"Updated rooting depth to {self.current_depth} mm at stage {growth_stage}.")


# Example Usage
if __name__ == "__main__":
    # Example crop and soil parameters
    crop_params = {"type": "corn", "growth_rate": 2.5}
    soil_params = {"type": "loam", "depth_limit": 1500}

    rooting_depth = RootingDepthFAO56(crop_params, soil_params)
    rooting_depth.initialize_rooting_depth(initial_depth=50)
    rooting_depth.update_rooting_depth(growth_stage="vegetative", growth_rate=3)