
from datetime import datetime, timedelta

class DateRange:
    def __init__(self, start_date, end_date):
        """Initialize the date range with start and end dates."""
        self.start = datetime.strptime(start_date, "%Y-%m-%d")
        self.end = datetime.strptime(end_date, "%Y-%m-%d")
        self.curr = self.start
        self.iDOY = self.curr.timetuple().tm_yday
        self.iDaysInMonth = (self.curr.replace(day=28) + timedelta(days=4)).day
        self.iDaysInYear = 366 if self.is_leap_year(self.curr.year) else 365
        self.iYearOfSimulation = 1
        self.lIsLeapYear = self.is_leap_year(self.curr.year)
        self.iNumDaysFromOrigin = 0
        self.iDayOfSimulation = 0

    def is_leap_year(self, year):
        """Determine if a given year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def initialize_datetimes(self, start_date, end_date):
        """Reinitialize the date range."""
        self.__init__(start_date, end_date)

    def days_from_origin(self):
        """Calculate the number of days from the start date to the current date."""
        return (self.curr - self.start).days

    def increment_by_one_day(self):
        """Increment the current date by one day."""
        self.curr += timedelta(days=1)
        self.iDOY = self.curr.timetuple().tm_yday
        self.iDayOfSimulation += 1

    def advance_to_last_day_of_year(self):
        """Advance the current date to the last day of the year."""
        self.curr = datetime(self.curr.year, 12, 31)
        self.iDOY = self.curr.timetuple().tm_yday

    def advance_to_last_day_of_month(self):
        """Advance the current date to the last day of the current month."""
        next_month = self.curr.replace(day=28) + timedelta(days=4)
        self.curr = next_month.replace(day=1) - timedelta(days=1)
        self.iDOY = self.curr.timetuple().tm_yday


# Example Usage
if __name__ == "__main__":
    dr = DateRange("2023-01-01", "2023-12-31")
    
    # Increment by one day
    dr.increment_by_one_day()
    print("Current Date after increment:", dr.curr)
    print("Day of Year:", dr.iDOY)
    
    # Advance to last day of the year
    dr.advance_to_last_day_of_year()
    print("Last Day of Year:", dr.curr)
    
    # Calculate days from origin
    print("Days from origin:", dr.days_from_origin())
