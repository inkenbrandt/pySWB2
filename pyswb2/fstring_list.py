
class FStringList:
    def __init__(self):
        self.s = []  # A list to hold strings
        self.count = 0  # Count of strings

    def add(self, string):
        """Add a string to the list."""
        self.s.append(string)
        self.count += 1

    def concatenate(self, other):
        """Concatenate another FStringList or string."""
        if isinstance(other, FStringList):
            self.s.extend(other.s)
        elif isinstance(other, str):
            self.s.append(other)
        else:
            raise TypeError("Unsupported type for concatenation.")
        self.count = len(self.s)

    def __str__(self):
        """Return all strings concatenated into one."""
        return ''.join(self.s)
