import numpy as np
import pandas as pd


# Simulating constants and helper functions
SIMULATION_NUMBER = None  # Placeholder for simulation number
RANDOM_START = 12345  # Random seed equivalent

# Reinitialize RANDOM_VALUES with dimensions matching iMaxRainZones and SIMULATION_NUMBER
iMaxRainZones = 101  # Example maximum rain zone from the simulated data
num_simulations = 5  # Number of simulations
RANDOM_VALUES = np.random.random((iMaxRainZones, num_simulations))  # Corrected dimensions

RAINFALL_ADJUST_FACTOR = None  # Placeholder for rainfall adjustment factors

# Placeholder for dictionary access, simulated as a Python dictionary
CF_DICT = {
    "FRAGMENTS_SEQUENCE_SIMULATION_NUMBER": [1],  # Example data
    "FRAGMENTS_DAILY_FILE": ["daily_fragments.txt"],
    "FRAGMENTS_SEQUENCE_FILE": ["<NA>"],
}

# Simulating core variables
RAINFALL_ZONE = {"data": np.random.random((10, 10))}  # Example rainfall zone data
FRAGMENTS = {"iRainGageZone": np.random.randint(1, 10, size=100)}  # Example fragment data

FRAGMENTS_SETS = {}  # Dictionary to store fragment sets keyed by rain gauge zone
FRAGMENTS_SEQUENCE = []  # List to store fragment sequence entries
# Reinitialize CURRENT_FRAGMENTS with dimensions matching iMaxRainZones and num_simulations
CURRENT_FRAGMENTS = np.empty((iMaxRainZones, num_simulations), dtype=object)  # Correct dimensions

RANDOM_VALUES = None  # Placeholder for random values (matrix)

# Translating the subroutine
def precipitation_method_of_fragments_initialize(lActive):
    global SIMULATION_NUMBER

    # Fetching simulation number
    iSimulationNumbers = CF_DICT.get("FRAGMENTS_SEQUENCE_SIMULATION_NUMBER", [])
    if iSimulationNumbers and iSimulationNumbers[0] > 0:
        SIMULATION_NUMBER = iSimulationNumbers[0]

    # Validate RAINFALL_ZONE exists
    if not RAINFALL_ZONE:
        raise ValueError("A RAINFALL_ZONE grid must be supplied in order to use this option.")

    # Map active cells
    active_cells_count = np.count_nonzero(lActive)
    RAIN_GAGE_ID = RAINFALL_ZONE["data"][lActive]

    # Allocate arrays
    RAINFALL_ADJUST_FACTOR = np.empty(active_cells_count)
    FRAGMENT_VALUE = np.empty(active_cells_count)

    # Read daily fragments
    fragments_file = CF_DICT["FRAGMENTS_DAILY_FILE"][0]
    read_daily_fragments(fragments_file)

    # Check and handle fragment sequence file
    fragments_sequence_file = CF_DICT["FRAGMENTS_SEQUENCE_FILE"][0]
    if fragments_sequence_file != "<NA>":
        read_fragments_sequence(fragments_sequence_file)

    # Process fragment sets
    iMaxRainZones = np.max(FRAGMENTS["iRainGageZone"])
    FRAGMENTS_SETS = np.empty(iMaxRainZones, dtype=object)

    # Allocate ancillary data structures
    CURRENT_FRAGMENTS = np.empty((iMaxRainZones, 1))
    RANDOM_VALUES = np.empty((iMaxRainZones, 1))

    # Initialize random generator if needed
    np.random.seed(RANDOM_START)

    process_fragment_sets()

    # Open and write fragments file
    with open("Fragments_as_implemented_by_SWB.csv", "w") as f:
        f.write("Simulation_Number, Month, Rain_Zone, Year, Random_Number, Fragment_Set,")
        f.write(",".join(["fragment"] * 30) + ",fragment\n")


# Simulate the structure of the FRAGMENTS data in Python
class Fragment:
    def __init__(self, month, rain_gage_zone, fragment_set, fragment_values):
        self.month = month
        self.rain_gage_zone = rain_gage_zone
        self.fragment_set = fragment_set
        self.fragment_values = fragment_values

# Function to normalize February fragments to ensure they sum to 1
def normalize_february_fragment_sequence(fragment):
    total = sum(fragment.fragment_values)
    if total != 0:
        fragment.fragment_values = [value / total for value in fragment.fragment_values]

def read_daily_fragments(filename):
    global FRAGMENTS

    # Open the file and process it line by line
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith(("#", "%", "!")):  # Skip comment lines
                continue
            parts = line.split()  # Split by whitespace (default delimiter)

            if len(parts) < 3:
                raise ValueError("Invalid file format. Expected at least month, rain gage zone, and fragment set.")

            # Parse month, rain gage zone, and fragment set
            month = int(parts[0])
            rain_gage_zone = int(parts[1])
            fragment_set = int(parts[2])

            # Parse daily fragment values (remaining columns)
            fragment_values = []
            for value in parts[3:]:
                fValue = float(value)
                if fValue < 0.0 or fValue > 1.0:
                    fValue = 0.0  # Substitute invalid values
                fragment_values.append(fValue)

            # Create and append the Fragment object
            fragment = Fragment(month, rain_gage_zone, fragment_set, fragment_values)
            FRAGMENTS.append(fragment)

            # Normalize February fragments if necessary
            if month == 2:
                normalize_february_fragment_sequence(fragment)

    print(f"Read {len(FRAGMENTS)} fragments from {filename}.")

# Define a class to represent each fragment set
class FragmentSet:
    def __init__(self, rain_gage_zone):
        self.rain_gage_zone = rain_gage_zone
        self.start_record = {month: None for month in range(1, 13)}
        self.number_of_fragments = {month: 0 for month in range(1, 13)}



def process_fragment_sets():
    global FRAGMENTS_SETS

    # Initialize variables
    iCount = 1
    iPreviousRainGageZone = FRAGMENTS[0].rain_gage_zone
    iPreviousMonth = FRAGMENTS[0].month

    # Initialize the first fragment set
    if iPreviousRainGageZone not in FRAGMENTS_SETS:
        FRAGMENTS_SETS[iPreviousRainGageZone] = FragmentSet(iPreviousRainGageZone)
    FRAGMENTS_SETS[iPreviousRainGageZone].start_record[iPreviousMonth] = 0

    # Iterate through fragments
    for iIndex in range(1, len(FRAGMENTS)):
        fragment = FRAGMENTS[iIndex]
        iRainGageZone = fragment.rain_gage_zone
        iMonth = fragment.month

        if iRainGageZone != iPreviousRainGageZone:
            # Finalize the previous zone and month
            FRAGMENTS_SETS[iPreviousRainGageZone].number_of_fragments[iPreviousMonth] = iCount

            # Initialize a new fragment set for the current zone
            if iRainGageZone not in FRAGMENTS_SETS:
                FRAGMENTS_SETS[iRainGageZone] = FragmentSet(iRainGageZone)
            FRAGMENTS_SETS[iRainGageZone].start_record[iMonth] = iIndex
            iCount = 1  # Reset counter
        else:
            iCount += 1

        # Update previous values
        iPreviousMonth = iMonth
        iPreviousRainGageZone = iRainGageZone

    # Finalize the last fragment set
    FRAGMENTS_SETS[iPreviousRainGageZone].number_of_fragments[iPreviousMonth] = iCount

    # Log summary of fragment sets
    print("### Summary of fragment sets in memory ###")
    print("gage number | month      | start index  | num records ")
    print("----------- | ---------- | ------------ | ------------")
    for zone, fragment_set in FRAGMENTS_SETS.items():
        for month in range(1, 13):
            start = fragment_set.start_record[month] or "N/A"
            count = fragment_set.number_of_fragments[month]
            print(f"{zone:11} | {month:10} | {start:12} | {count:12}")


# Define a class to represent each fragment sequence entry
class FragmentSequence:
    def __init__(self, sim_number, sim_month, sim_rainfall_zone, sim_year, sim_random_number, sim_selected_set):
        self.sim_number = sim_number
        self.sim_month = sim_month
        self.sim_rainfall_zone = sim_rainfall_zone
        self.sim_year = sim_year
        self.sim_random_number = sim_random_number
        self.sim_selected_set = sim_selected_set



def read_fragments_sequence(filename):
    global FRAGMENTS_SEQUENCE, CURRENT_FRAGMENTS, RANDOM_VALUES

    # Read the file using pandas
    data = pd.read_csv(
        filename,
        comment="#",  # Skip comments
        delim_whitespace=True,  # Handle whitespace-delimited format
        header=0,  # Assume first row is a header
        names=["sim_number", "sim_month", "sim_rainfall_zone", "sim_year", "sim_random_number", "sim_selected_set"],
    )

    # Parse and validate the data
    for _, row in data.iterrows():
        try:
            fragment_seq = FragmentSequence(
                sim_number=int(row["sim_number"]),
                sim_month=int(row["sim_month"]),
                sim_rainfall_zone=int(row["sim_rainfall_zone"]),
                sim_year=int(row["sim_year"]),
                sim_random_number=float(row["sim_random_number"]),
                sim_selected_set=int(row["sim_selected_set"]),
            )
            FRAGMENTS_SEQUENCE.append(fragment_seq)
        except ValueError as e:
            print(f"Error parsing line: {row}. Error: {e}")
            continue

    # Allocate placeholders for CURRENT_FRAGMENTS and RANDOM_VALUES
    max_rain_gage_number = max(seq.sim_rainfall_zone for seq in FRAGMENTS_SEQUENCE)
    max_simulation_number = max(seq.sim_number for seq in FRAGMENTS_SEQUENCE)
    CURRENT_FRAGMENTS = np.empty((max_rain_gage_number, max_simulation_number), dtype=object)
    RANDOM_VALUES = np.random.random((max_rain_gage_number, max_simulation_number))  # Simulated random values

    # Logging
    print("### Summary of fragment sequence sets in memory ###")
    print("sim number | rainfall zone | month  | year   | selected set")
    print("-----------|---------------|--------|--------|-------------")
    for seq in FRAGMENTS_SEQUENCE:
        print(f"{seq.sim_number:<11}|{seq.sim_rainfall_zone:<15}|{seq.sim_month:<8}|{seq.sim_year:<8}|{seq.sim_selected_set:<13}")


def normalize_february_fragment_sequence(iCount):
    """
    Normalize February fragment values to ensure the first 28 days sum to 1.
    """
    fragment = FRAGMENTS[iCount]

    # Check for leap year fragment (day 29)
    if fragment.fragment_values[28] > 0:  # Day 29 (index 28 in Python)
        # Calculate the sum of the first 28 days
        sum_fragments = sum(fragment.fragment_values[:28])
        if sum_fragments > 0:
            # Normalize the first 28 days
            fragment.fragment_values[:28] = [v / sum_fragments for v in fragment.fragment_values[:28]]

        # Zero out days 29–31
        fragment.fragment_values[28:] = [0] * (len(fragment.fragment_values) - 28)

def update_random_values():
    """
    Updates the RANDOM_VALUES array with new random numbers.
    """
    global RANDOM_VALUES
    RANDOM_VALUES = np.random.random(RANDOM_VALUES.shape)


def update_fragments(lShuffle, current_date, simulation_number):
    """
    Update fragment values based on simulation parameters.

    Args:
    - lShuffle (bool): Whether to shuffle the fragment selection.
    - current_date (dict): Dictionary with 'month' and 'day' keys.
    - simulation_number (int): Current simulation number.
    """
    global CURRENT_FRAGMENTS, RANDOM_VALUES, FRAGMENTS

    # Initialize
    iMaxRainZones = max(fragment.rain_gage_zone for fragment in FRAGMENTS)
    iMonth = current_date["month"]
    iDay = current_date["day"]

    # Default fragment values
    FRAGMENT_VALUE = np.zeros(len(FRAGMENTS))

    for rain_zone in range(1, iMaxRainZones + 1):
        if lShuffle:
            # Update random values
            update_random_values()

            # Find target record
            fragment_set = FRAGMENTS_SETS.get(rain_zone)
            if not fragment_set:
                continue

            iStartRecord = fragment_set.start_record.get(iMonth, 0)
            iNumberOfFragments = fragment_set.number_of_fragments.get(iMonth, 0)
            iEndRecord = iStartRecord + iNumberOfFragments - 1

            # Select a random fragment
            random_value = RANDOM_VALUES[rain_zone - 1, simulation_number - 1]
            iTargetRecord = iStartRecord + int(random_value * iNumberOfFragments)

            if iTargetRecord < 0 or iTargetRecord >= len(FRAGMENTS):
                print(f"Error: Target record {iTargetRecord} out of bounds for rain zone {rain_zone}.")
                continue

            # Assign selected fragment to CURRENT_FRAGMENTS
            CURRENT_FRAGMENTS[rain_zone - 1, simulation_number - 1] = FRAGMENTS[iTargetRecord]

            # Logging (Simulated)
            selected_fragment = FRAGMENTS[iTargetRecord]
            print(
                f"Simulation {simulation_number}, Month {selected_fragment.month}, "
                f"Zone {selected_fragment.rain_gage_zone}, Selected Set {selected_fragment.fragment_set}"
            )

        # Assign fragment value to cells matching the current rain zone
        for i, fragment in enumerate(FRAGMENTS):
            if fragment.rain_gage_zone == rain_zone:
                FRAGMENT_VALUE[i] = CURRENT_FRAGMENTS[rain_zone - 1, simulation_number - 1].fragment_values[iDay - 1]

    return FRAGMENT_VALUE


def precipitation_method_of_fragments_calculate(lActive, current_date, first_call=True):
    """
    Calculate synthetic precipitation using the method of fragments.

    Args:
    - lActive (numpy.ndarray): Logical array indicating active cells.
    - current_date (dict): Dictionary with 'day' and 'month' keys.
    - first_call (bool): Indicates if it's the first call.
    """
    global RAINFALL_ADJUST_FACTOR

    # Check if it's the first day of the month or the first function call
    if current_date["day"] == 1 or first_call:
        # Locate and retrieve the rainfall adjustment factor grid
        # Simulating data retrieval
        pRAINFALL_ADJUST_FACTOR = np.random.random(lActive.shape)  # Simulated 2D grid
        RAINFALL_ADJUST_FACTOR = pRAINFALL_ADJUST_FACTOR[lActive]

        # Update fragments with shuffling enabled
        update_fragments(lShuffle=True, current_date=current_date, simulation_number=SIMULATION_NUMBER)

        first_call = False  # Update the first call flag
    else:
        # Update fragments without shuffling
        update_fragments(lShuffle=False, current_date=current_date, simulation_number=SIMULATION_NUMBER)

    # Returning rainfall adjustment factors and updated fragments for verification
    return RAINFALL_ADJUST_FACTOR


