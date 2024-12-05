
import numpy as np

class NetCDFFile:
    def __init__(self, file_path=None):
        self.file_path = file_path

class NetCDFFileCollection:
    def __init__(self):
        self.ncfile = None

class OutputSpecs:
    def __init__(self, variable_name, variable_units, valid_minimum, valid_maximum, is_active, multisim):
        self.variable_name = variable_name
        self.variable_units = variable_units
        self.valid_minimum = valid_minimum
        self.valid_maximum = valid_maximum
        self.is_active = is_active
        self.multisim = multisim

class OutputModule:
    def __init__(self):
        self.RECHARGE_ARRAY = None
        self.NC_OUT = []
        self.NC_MULTI_SIM_OUT = []
        self.NCDF_NUM_OUTPUTS = 29

    def initialize_recharge_array(self, size):
        self.RECHARGE_ARRAY = np.zeros(size, dtype=float)

    def add_output_spec(self, variable_name, variable_units, valid_minimum, valid_maximum, is_active, multisim):
        spec = OutputSpecs(variable_name, variable_units, valid_minimum, valid_maximum, is_active, multisim)
        return spec

    def setup_netCDF_output(self, num_files):
        self.NC_OUT = [NetCDFFileCollection() for _ in range(num_files)]

    def setup_multisim_output(self, num_rows, num_cols):
        self.NC_MULTI_SIM_OUT = [[NetCDFFileCollection() for _ in range(num_cols)] for _ in range(num_rows)]
