class ASCIIFile:
    """
    Python translation of the Fortran type ASCII_FILE_T.
    Represents an ASCII file with attributes and methods for file operations.
    """
    def __init__(self, filename, delimiters=",", comment_chars="#", header_lines=1):
        """
        Initializes the ASCII file structure.

        Parameters:
        - filename (str): Path to the file.
        - delimiters (str): Delimiters used in the file (default: ",").
        - comment_chars (str): Characters indicating comments (default: "#").
        - header_lines (int): Number of header lines in the file (default: 1).
        """
        self.filename = filename
        self.delimiters = delimiters
        self.comment_chars = comment_chars
        self.header_lines = header_lines
        self.column_names = []  # List of column names (populated after reading the header)
        self.data = []  # List of records (populated after reading the file)
        self.is_open = False  # Status flag

    def open_file(self):
        """
        Opens the file and reads its contents into memory.
        """
        try:
            with open(self.filename, 'r') as file:
                self.is_open = True
                lines = file.readlines()

                # Process header lines
                for i in range(self.header_lines):
                    header_line = lines[i].strip()
                    if not header_line.startswith(self.comment_chars):
                        self.column_names = header_line.split(self.delimiters)

                # Process data lines
                for line in lines[self.header_lines:]:
                    if not line.strip().startswith(self.comment_chars):
                        self.data.append(line.strip().split(self.delimiters))
        except FileNotFoundError:
            self.is_open = False
            raise FileNotFoundError(f"File {self.filename} not found.")
        except Exception as e:
            self.is_open = False
            raise IOError(f"An error occurred while reading the file: {e}")

    def write_file(self, output_filename):
        """
        Writes the current data to a new file.

        Parameters:
        - output_filename (str): Path to the output file.
        """
        try:
            with open(output_filename, 'w') as file:
                # Write column names as header
                file.write(self.delimiters.join(self.column_names) + "\n")
                # Write data records
                for record in self.data:
                    file.write(self.delimiters.join(record) + "\n")
        except Exception as e:
            raise IOError(f"An error occurred while writing the file: {e}")

    def display_content(self):
        """
        Displays the content of the file for debugging purposes.
        """
        print("Column Names:", self.column_names)
        print("Data:")
        for record in self.data:
            print(record)

if __name__ == "__main__":
    # Example usage
    ascii_file = ASCIIFile(filename="example.csv", delimiters=",", comment_chars="#", header_lines=1)

    # Attempt to open and display the file content
    try:
        ascii_file.open_file()
        ascii_file.display_content()
    except FileNotFoundError:
        print("The specified file does not exist.")

    # Example: Writing data to a new file
    ascii_file.write_file("output.csv")
