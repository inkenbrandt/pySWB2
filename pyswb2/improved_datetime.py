
import pandas as pd

class DateTime:
    """Class for handling date and time operations using pandas."""

    def __init__(self, year, month, day):
        """Initialize the DateTime object.

        Args:
            year (int): Year value.
            month (int): Month value.
            day (int): Day value.
        """
        self.timestamp = pd.Timestamp(year=year, month=month, day=day)

    def is_leap_year(self):
        """Check if the year is a leap year.

        Returns:
            bool: True if leap year, False otherwise.
        """
        return self.timestamp.is_leap_year

    def day_of_year(self):
        """Calculate the day of the year.

        Returns:
            int: The day of the year.
        """
        return self.timestamp.day_of_year

    def julian_day(self):
        """Convert to Julian day.

        Returns:
            int: Julian day number.
        """
        return self.timestamp.to_julian_date()

    @staticmethod
    def from_julian_day(julian_day):
        """Convert Julian day to Gregorian date.

        Args:
            julian_day (float): Julian day number.

        Returns:
            DateTime: Corresponding DateTime object.
        """
        timestamp = pd.Timestamp.from_julian_date(julian_day)
        return DateTime(timestamp.year, timestamp.month, timestamp.day)

    def __gt__(self, other):
        """Check if this date is greater than another.

        Args:
            other (DateTime): Another DateTime object.

        Returns:
            bool: True if this date is greater.
        """
        return self.timestamp > other.timestamp

    def __lt__(self, other):
        """Check if this date is less than another.

        Args:
            other (DateTime): Another DateTime object.

        Returns:
            bool: True if this date is less.
        """
        return self.timestamp < other.timestamp
