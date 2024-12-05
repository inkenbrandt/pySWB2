
class Proj4Support:
    def __init__(self):
        pass

    def create_attributes_from_proj4_string(self, proj4_string):
        """Parse a PROJ4 string into attribute names and values."""
        attribute_name_list = []
        attribute_value_list = []

        # Split the proj4 string by spaces and process each parameter
        proj4_params = proj4_string.split()
        for param in proj4_params:
            if '=' in param:
                name, value = param.split('=', 1)
                attribute_name_list.append(name.strip())
                attribute_value_list.append(value.strip())
            else:
                # Parameters without values
                attribute_name_list.append(param.strip())
                attribute_value_list.append(None)

        return attribute_name_list, attribute_value_list


# Example Usage
if __name__ == "__main__":
    proj4_support = Proj4Support()
    proj4_string = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"
    names, values = proj4_support.create_attributes_from_proj4_string(proj4_string)
    print("Attribute Names:", names)
    print("Attribute Values:", values)
