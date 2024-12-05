
class DictEntry:
    """Represents an entry in the dictionary."""

    def __init__(self, key=None, secondary_key=None):
        """Initialize a dictionary entry.

        Args:
            key (str): Primary key for the entry.
            secondary_key (str): Secondary key for the entry.
        """
        self.key = key
        self.secondary_key = secondary_key
        self.values = []  # Holds various types of values (e.g., strings, numbers)
        self.previous = None  # Pointer to the previous entry
        self.next = None  # Pointer to the next entry

    def add_value(self, value):
        """Add a value to the entry.

        Args:
            value: Value to add (e.g., string, int, float, bool).
        """
        self.values.append(value)


class Dictionary:
    """Represents a linked-list-based dictionary."""

    def __init__(self):
        """Initialize the dictionary."""
        self.head = None
        self.tail = None

    def add_entry(self, key, secondary_key=None):
        """Add a new dictionary entry.

        Args:
            key (str): Primary key for the entry.
            secondary_key (str): Secondary key for the entry (optional).

        Returns:
            DictEntry: The created dictionary entry.
        """
        new_entry = DictEntry(key, secondary_key)
        if self.tail:
            self.tail.next = new_entry
            new_entry.previous = self.tail
        self.tail = new_entry
        if not self.head:
            self.head = new_entry
        return new_entry

    def find_entry(self, key):
        """Find an entry by its primary key.

        Args:
            key (str): The key to search for.

        Returns:
            DictEntry: The found entry, or None if not found.
        """
        current = self.head
        while current:
            if current.key == key:
                return current
            current = current.next
        return None
