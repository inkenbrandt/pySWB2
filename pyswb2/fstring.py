
class FString:
    def __init__(self, content=""):
        self.content = str(content)  # Store content as a string

    def __add__(self, other):
        """Overload the + operator for concatenation."""
        if isinstance(other, FString):
            return FString(self.content + other.content)
        else:
            return FString(self.content + str(other))

    def __eq__(self, other):
        """Overload the = operator for comparison."""
        return self.content == str(other)

    def __str__(self):
        """Return the string representation."""
        return self.content
