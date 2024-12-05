import rasterio
import numpy as np
import os
from pathlib import Path


def tiff_to_ascii(input_tiff, output_ascii=None, no_data_value=-9999):
    """
    Convert a GeoTIFF file to ASCII grid format.

    Parameters:
    input_tiff (str): Path to input GeoTIFF file
    output_ascii (str): Path to output ASCII file (optional, will derive from input if not provided)
    no_data_value (int/float): Value to use for no data pixels

    Returns:
    str: Path to the created ASCII file
    """
    # If no output path specified, create one from input path
    if output_ascii is None:
        output_ascii = str(Path(input_tiff).with_suffix('.asc'))

    try:
        with rasterio.open(input_tiff) as src:
            # Read the data and mask
            data = src.read(1)
            if src.nodata is not None:
                data = np.where(data == src.nodata, no_data_value, data)

            # Get the geometric properties
            transform = src.transform
            cellsize_x = transform[0]
            cellsize_y = abs(transform[4])  # Make sure cellsize is positive

            # Get upper left corner coordinates
            xllcorner = transform[2]
            yllcorner = transform[5] + (src.height * transform[4])

            # Write ASCII file
            with open(output_ascii, 'w') as f:
                # Write header
                f.write(f"ncols {src.width}\n")
                f.write(f"nrows {src.height}\n")
                f.write(f"xllcorner {xllcorner}\n")
                f.write(f"yllcorner {yllcorner}\n")
                f.write(f"cellsize {cellsize_x}\n")  # Assuming square pixels
                f.write(f"NODATA_value {no_data_value}\n")

                # Write data
                for row in data:
                    row_str = ' '.join(map(str, row))
                    f.write(f"{row_str}\n")

        print(f"Successfully converted {input_tiff} to {output_ascii}")
        return output_ascii

    except Exception as e:
        print(f"Error converting {input_tiff}: {str(e)}")
        return None


def batch_convert_tiffs(input_folder, output_folder=None, pattern="*.tif*"):
    """
    Convert all GeoTIFF files in a folder to ASCII format.

    Parameters:
    input_folder (str): Path to folder containing GeoTIFF files
    output_folder (str): Path to output folder (optional)
    pattern (str): File pattern to match (default: "*.tif*")

    Returns:
    list: Paths to created ASCII files
    """
    input_path = Path(input_folder)

    # If no output folder specified, create in same location as input
    if output_folder is None:
        output_folder = input_path
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    converted_files = []

    # Process each TIFF file
    for tiff_file in input_path.glob(pattern):
        output_ascii = output_folder / tiff_file.with_suffix('.asc').name
        result = tiff_to_ascii(str(tiff_file), str(output_ascii))
        if result:
            converted_files.append(result)

    return converted_files


# Example usage
if __name__ == "__main__":
    # Convert a single file
    tiff_to_ascii("dc_nlcd_wms.tiff")

    # Or convert all TIFFs in a folder
    batch_convert_tiffs("./nlcd_data")