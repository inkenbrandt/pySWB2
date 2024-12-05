
class SummaryStatistics:
    def __init__(self):
        self.poly_id = []  # List to store polygon IDs
        self.landuse_codes = []  # List to store land use codes
        self.anything_to_summarize = False  # Flag to indicate if there is data to summarize

    def add_polygon_id_to_summary_list(self, polygon_id):
        """Add a polygon ID to the summary list."""
        self.poly_id.append(polygon_id)
        self.anything_to_summarize = True

    def add_landuse_code_to_summary_list(self, landuse_code):
        """Add a land use code to the summary list."""
        self.landuse_codes.append(landuse_code)
        self.anything_to_summarize = True

    def summarize(self):
        """Generate a summary of the stored data."""
        if not self.anything_to_summarize:
            return "No data to summarize."

        summary = {
            "Polygon Count": len(self.poly_id),
            "Land Use Code Count": len(self.landuse_codes),
            "Unique Land Use Codes": len(set(self.landuse_codes))
        }
        return summary


# Example Usage
if __name__ == "__main__":
    stats = SummaryStatistics()

    # Adding some data
    stats.add_polygon_id_to_summary_list("Poly_001")
    stats.add_polygon_id_to_summary_list("Poly_002")
    stats.add_landuse_code_to_summary_list("LU_01")
    stats.add_landuse_code_to_summary_list("LU_02")
    stats.add_landuse_code_to_summary_list("LU_01")  # Duplicate for demonstration

    # Print summary
    print("Summary:")
    print(stats.summarize())
