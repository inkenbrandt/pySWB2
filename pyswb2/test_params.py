
class TestParams:
    def __init__(self):
        self.params = {}  # Simulating PARAMETERS_T with a dictionary

    def add_file(self, file_name, delimiters=None):
        """Add a parameter file to the configuration."""
        self.params[file_name] = {"delimiters": delimiters}
        print(f"Added file: {file_name} with delimiters: {delimiters}")

    def show_params(self):
        """Display all parameter files in the configuration."""
        return self.params


# Example Usage
if __name__ == "__main__":
    test = TestParams()
    test.add_file("LU_lookup_NLCD.txt")
    test.add_file("rain_adj_factors_maui.prn", delimiters="WHITESPACE")
    print("Current Parameters:")
    print(test.show_params())
