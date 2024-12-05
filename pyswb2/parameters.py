class Parameters:
    """
    A class for parameter storage and retrieval.
    """

    MAX_TABLE_RECORD_LEN = 2048  # Equivalent constant from FORTRAN

    def __init__(self):
        self.filenames = []
        self.delimiters = []
        self.comment_chars = []
        self.parameters_dict = {}
        self.count = 0

    def add_file(self, filename, delimiter=None, comment_chars=None):
        """
        Adds a filename and optional metadata.
        """
        self.filenames.append(filename)
        self.delimiters.append(delimiter if delimiter else ",")
        self.comment_chars.append(comment_chars if comment_chars else "#")
        self.count += 1

    def parse_files(self):
        """
        Parse the added files to populate the parameters dictionary.
        Handles delimiters and skips comments.
        """
        for filename, delimiter, comment_char in zip(self.filenames, self.delimiters, self.comment_chars):
            with open(filename, 'r') as file:
                for line in file:
                    if not line.startswith(comment_char):  # Skip comments
                        key, value = line.strip().split(delimiter)
                        self.parameters_dict[key] = value

    def get_parameter(self, key, data_type=str):
        """
        Retrieve a parameter and cast it to the specified type.
        """
        if key not in self.parameters_dict:
            raise KeyError(f"Parameter '{key}' not found.")
        return data_type(self.parameters_dict[key])


if __name__ == "__main__":
    # Globals
    PARAMS = Parameters()  # Equivalent to FORTRAN's global `PARAMS`

    # Displaying the generated Python code
    PARAMS.MAX_TABLE_RECORD_LEN
