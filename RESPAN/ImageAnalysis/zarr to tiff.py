import os
import numpy as np
import zarr
import tifffile
import dask.array as da
from pathlib import Path
import argparse
import time
import psutil
import gc

def zarr_to_tiff(zarr_path, tiff_path, chunk_size=None, compression='zlib'):
    """
    Convert a Zarr array to a TIFF file with efficient memory handling.

    Parameters:
    -----------
    zarr_path : str
        Path to the input Zarr directory
    tiff_path : str
        Path to the output TIFF file
    chunk_size : int, optional
        Z-dimension chunk size for processing (default: auto-calculated based on RAM)
    compression : str, optional
        TIFF compression method ('zlib', 'lzw', 'jpeg', etc.)
    """
    print(f"Opening Zarr from: {zarr_path}")
    start_time = time.time()

    # Check if zarr path exists and has expected structure
    zarr_dir = Path(zarr_path)
    if not zarr_dir.exists():
        raise FileNotFoundError(f"Zarr directory not found: {zarr_path}")

    # Try to open the zarr
    is_ome_zarr = False
    try:
        # Check for OME-Zarr structure (looking for '0' component)
        if (zarr_dir / '0').exists():
            zarr_array = zarr.open(str(zarr_dir / '0'), mode='r')
            print(f"Opened OME-Zarr component: 0")
            is_ome_zarr = True
        else:
            zarr_array = zarr.open(str(zarr_dir), mode='r')
    except Exception as e:
        raise ValueError(f"Error opening Zarr: {str(e)}")

    # Get array info
    shape = zarr_array.shape
    dtype = zarr_array.dtype
    ndim = len(shape)

    print(f"Zarr info: Shape={shape}, Dtype={dtype}, Ndim={ndim}")

    # Validate dimensions (expect CZYX or ZYX)
    if ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D array (ZYX or CZYX), got {ndim}D")

    # Calculate memory requirements and determine optimal chunk size
    if chunk_size is None:
        # Use at most 25% of available RAM for a single chunk
        available_ram = psutil.virtual_memory().available * 0.25
        single_slice_bytes = np.prod(shape[-2:]) * np.dtype(dtype).itemsize

        if ndim == 4:  # CZYX
            single_slice_bytes *= shape[0]  # Account for channels
            z_dim = shape[1]
        else:  # ZYX
            z_dim = shape[0]

        # Calculate how many Z slices we can process at once
        chunk_size = max(1, int(available_ram / single_slice_bytes))
        chunk_size = min(chunk_size, z_dim)  # Don't exceed total Z size

    print(f"Using chunk size of {chunk_size} for Z dimension")

    # Create a dask array pointing to the zarr
    if is_ome_zarr:
        dask_array = da.from_zarr(str(zarr_dir / '0'))
    else:
        dask_array = da.from_zarr(str(zarr_dir))

    # Create metadata for TIFF
    metadata = {
        'axes': 'CZYX' if ndim == 4 else 'ZYX',
        'spacing': zarr_array.attrs.get('pixelSize', [1.0, 1.0, 1.0]),
        'unit': 'um'
    }

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(tiff_path)), exist_ok=True)

    # Process and save by chunks
    if ndim == 4:  # CZYX
        z_dim = shape[1]

        # Create ImageJ-compatible hyperstack TIFF
        with tifffile.TiffWriter(tiff_path, imagej=True, bigtiff=True) as tif:
            for z_start in range(0, z_dim, chunk_size):
                z_end = min(z_start + chunk_size, z_dim)
                print(f"Processing Z slices {z_start}-{z_end} of {z_dim} ({(z_end - z_start) / z_dim * 100:.1f}%)")

                # Load chunk into memory
                chunk = dask_array[:, z_start:z_end, :, :].compute()

                # Save to TIFF (append mode)
                if z_start == 0:
                    tif.write(
                        chunk,
                        compression=compression,
                        metadata=metadata,
                        photometric='minisblack'
                    )
                else:
                    tif.write(chunk, compression=compression, photometric='minisblack')

                # Clean up
                del chunk
                gc.collect()

    else:  # ZYX
        z_dim = shape[0]

        # Create ImageJ-compatible TIFF
        with tifffile.TiffWriter(tiff_path, imagej=True, bigtiff=True) as tif:
            for z_start in range(0, z_dim, chunk_size):
                z_end = min(z_start + chunk_size, z_dim)
                print(f"Processing Z slices {z_start}-{z_end} of {z_dim} ({(z_end - z_start) / z_dim * 100:.1f}%)")

                # Load chunk into memory
                chunk = dask_array[z_start:z_end, :, :].compute()

                # Save to TIFF (append mode)
                if z_start == 0:
                    tif.write(
                        chunk,
                        compression=compression,
                        metadata=metadata,
                        photometric='minisblack'
                    )
                else:
                    tif.write(chunk, compression=compression, photometric='minisblack')

                # Clean up
                del chunk
                gc.collect()

    elapsed_time = time.time() - start_time
    file_size_gb = os.path.getsize(tiff_path) / (1024 ** 3)

    print(f"Conversion completed in {elapsed_time:.2f} seconds")
    print(f"Output file: {tiff_path} ({file_size_gb:.2f} GB)")


if __name__ == "__main__":
    zarr_path = r"D:\Project_Data\RESPAN\Testing\_2024_08_Test_with_Spines\1\Image13.ome.zarr"
    tiff_path = r"D:\Project_Data\RESPAN\Testing\_2024_08_Test_with_Spines\1\Image13.ome.zarr\output.tiff"

    zarr_to_tiff(zarr_path, tiff_path)