import os
import sys
import io

import time

import zipfile
import pydicom
import git

import numpy as np

import scipy.interpolate
import numba_interpolate

import skimage.measure 

import nrrd

c_out_pixel_spacing = np.array((2.23214293, 2.23214293, 3.))
c_resample_tolerance = 0.01 # Only interpolate voxels further off of the voxel grid than this

c_interpolate_seams = False # If yes, cut overlaps between stations to at most c_max_overlap and interpolate along them, otherwise cut at center of overlap
c_correct_intensity = False # If yes, apply intensity correction along overlap
c_max_overlap = 8 # Used in interpolation, any station overlaps are cut to be most this many voxels in size

c_trim_axial_slices = 4 # Trim this many axial slices from the output volume to remove folding artefacts

c_store_signals = True # If yes, store signal images

c_store_fractions = False # If yes, calculate fat and water fraction by station and fuse the result. The resulting images can not necessarily be calculated from the signal images directly
c_mask_fractions = True # If yes, attempt to remove background noise from the fraction images
c_mask_ratio = 0.1 # When creating fraction images, mask out voxels darker than this ratio of the total range of intensities

c_store_nrrd = True
c_store_mip = False
c_mip_encode_fraction = True # When writing mips, normalize the water or fat fractions

c_datatype_numpy = "float32" # See: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
c_datatype_nrrd = "float"    # See: https://github.com/mhe/pynrrd/blob/master/nrrd/reader.py

c_use_gpu = True # If yes, use numba for gpu access, otherwise use scipy on cpu


def dicomToVolume(input_path_zip):

    if not os.path.exists(input_path_zip):
        print("Could not find input file {}".format(input_path_zip))
        return

    # Extract imaging stations with water signal
    (voxel_data_w, positions_w, pixel_spacings, timestamps_w) = stationsFromDicom(input_path_zip, "_W")

    # Remove repeated stations
    (voxel_data_w, positions, pixel_spacings) = ensureStationConsistency(voxel_data_w, positions_w, timestamps_w, pixel_spacings)

    #
    volumes_out_w = []
    headers = []
    
    # Only use abdominal imaging stations
    for i in range(1, 3):

        #
        voxel_data_w[i] = np.flip(voxel_data_w[i], 2)
        voxel_data_w[i] = np.swapaxes(voxel_data_w[i], 0, 1)

        #
        origin_i = np.array(positions[i][[1, 0, 2]])

        #
        volumes_out_w.append(voxel_data_w[i])
        headers.append(formatNrrdHeader(voxel_data_w[i], pixel_spacings[i], origin_i))
        
    return (volumes_out_w, headers)


def formatNrrdHeader(volume, pixel_spacing, origin):

    header = {'dimension': 3}
    header['type'] = c_datatype_nrrd
    header['sizes'] = volume.shape
    header['pipeline_version'] = "blargh"

    # Spacing info compatible with 3D Slicer
    header['space dimension'] = 3
    header['space directions'] = np.array(pixel_spacing * np.eye(3,3))
    header['space origin'] = origin
    header['space units'] = "\"mm\" \"mm\" \"mm\""

    header['encoding'] = 'gzip'

    return header


def ensureStationConsistency(voxel_data_w, positions_w, timestamps_w, pixel_spacings):

    # In case of redundant stations, choose the newest
    if len(np.unique(positions_w, axis=0)) != len(positions_w):

        seg_select = []

        for pos in np.unique(positions_w, axis=0):

            # Find stations at current position
            offsets = np.array(positions_w) - np.tile(pos, (len(positions_w), 1))
            dist = np.sum(np.abs(offsets), axis=1)

            indices_p = np.where(dist == 0)[0]

            if len(indices_p) > 1:

                # Choose newest station
                timestamps_w_p = [str(x).replace(".", "") for f, x in enumerate(timestamps_w) if f in indices_p]

                # If you get scanned around midnight its your own fault
                recent_p = np.argmax(np.array(timestamps_w_p))

                print("WARNING: Image stations ({}) are superimposed. Choosing most recently imaged one ({})".format(indices_p, indices_p[recent_p]))
                
                seg_select.append(indices_p[recent_p])
            else:
                seg_select.append(indices_p[0])
        
        voxel_data_w = [x for f,x in enumerate(voxel_data_w) if f in seg_select]        
        positions_w = [x for f,x in enumerate(positions_w) if f in seg_select]        
        timestamps_w = [x for f,x in enumerate(timestamps_w) if f in seg_select]        
        pixel_spacings = [x for f,x in enumerate(pixel_spacings) if f in seg_select]        

    # Sort by position
    pos_z = np.array(positions_w)[:, 2]
    (pos_z, pos_indices) = zip(*sorted(zip(pos_z, np.arange(len(pos_z))), reverse=True))

    voxel_data_w = [voxel_data_w[i] for i in pos_indices]
    positions_w = [positions_w[i] for i in pos_indices]
    timestamps_w = [timestamps_w[i] for i in pos_indices]

    pixel_spacings = [pixel_spacings[i] for i in pos_indices]

    return (voxel_data_w, positions_w, pixel_spacings)


def extractStationsForModality(tag, station_names, station_voxel_data, station_positions, station_pixel_spacings, station_timestamps):

    # Merge all stations with given tag
    indices_t = [f for f, x in enumerate(station_names) if str(tag) in str(x)]

    voxel_data_t = [x for f, x in enumerate(station_voxel_data) if f in indices_t]
    positions_t = [x for f, x in enumerate(station_positions) if f in indices_t]
    pixel_spacings_t = [x for f, x in enumerate(station_pixel_spacings) if f in indices_t]
    timestamps_t = [x for f, x in enumerate(station_timestamps) if f in indices_t]
    
    return (voxel_data_t, positions_t, pixel_spacings_t, timestamps_t)


def getSignalSliceNamesInZip(z):

    file_names = [f.filename for f in z.infolist()]

    # Search for manifest file (name may be misspelled)
    csv_name = [f for f in file_names if "manifest" in f][0]

    with z.open(csv_name) as f0:

        data = f0.read() # Decompress into memory

        entries = str(data).split("\\n")
        entries.pop(-1)

        # Remove trailing blank lines
        entries = [f for f in entries if f != ""]

        # Get indices of relevant columns
        header_elements = entries[0].split(",")
        column_filename = [f for f,x in enumerate(header_elements) if "filename" in x][0]

        # Search for tags such as "Dixon_noBH_F". The manifest header can not be relied on
        for e in entries:
            entry_parts = e.split(",")
            column_desc = [f for f,x in enumerate(entry_parts) if "Dixon_noBH_F" in x]

            if column_desc:
                column_desc = column_desc[0]
                break

        # Get slice descriptions and filenames
        descriptions = [f.split(",")[column_desc] for f in entries]
        filenames = [f.split(",")[column_filename] for f in entries]

        # Extract signal images only
        chosen_rows = [f for f,x in enumerate(descriptions) if "_W" in x or "_F" in x]
        chosen_filenames = [x for f,x in enumerate(filenames) if f in chosen_rows]

    return chosen_filenames


def groupSlicesToStations(slice_pixel_data, slice_series, slice_names, slice_positions, slice_pixel_spacings, slice_times):

    # Group by series into stations
    unique_series = np.unique(slice_series)

    #
    station_voxel_data = []
    station_series = []
    station_names = []
    station_positions = []
    station_voxel_spacings = []
    station_times = []

    # Each series forms one station
    for s in unique_series:

        # Get slice indices for series s
        indices_s = [f for f, x in enumerate(slice_series) if x == s]

        # Get physical positions of slices
        slice_positions_s = [x for f, x in enumerate(slice_positions) if f in indices_s]

        position_max = np.amax(np.array(slice_positions_s).astype("float"), axis=0)
        station_positions.append(position_max)

        # Combine slices to station
        voxel_data_s = slicesToStationData(indices_s, slice_positions_s, slice_pixel_data)
        station_voxel_data.append(voxel_data_s)

        # Get index of first slice
        slice_0 = indices_s[0]

        station_series.append(slice_series[slice_0])
        station_names.append(slice_names[slice_0])
        station_times.append(slice_times[slice_0])

        # Get 3d voxel spacing
        voxel_spacing_2d = slice_pixel_spacings[slice_0]

        # Get third dimension by dividing station extent by slice count
        z_min = np.amin(np.array(slice_positions_s)[:, 2].astype("float"))
        z_max = np.amax(np.array(slice_positions_s)[:, 2].astype("float"))
        z_spacing = (z_max - z_min) / (len(slice_positions_s) - 1)

        voxel_spacing = np.hstack((voxel_spacing_2d, z_spacing))
        station_voxel_spacings.append(voxel_spacing)

    return (station_voxel_data, station_names, station_positions, station_voxel_spacings, station_times)


def getDataFromDicom(ds):

    pixel_data = ds.pixel_array

    series = ds.get_item(["0020", "0011"]).value
    series = int(series)

    position = ds.get_item(["0020", "0032"]).value 
    position = np.array(position.decode().split("\\")).astype("float32")

    pixel_spacing = ds.get_item(["0028", "0030"]).value
    pixel_spacing = np.array(pixel_spacing.decode().split("\\")).astype("float32")

    start_time = ds.get_item(["0008", "0031"]).value

    return (pixel_data, series, position, pixel_spacing, start_time)


def slicesToStationData(slice_indices, slice_positions, slices):

    # Get size of output volume station
    slice_count = len(slice_indices)
    slice_shape = slices[slice_indices[0]].shape

    # Get slice positions
    slices_z = np.zeros(slice_count)
    for z in range(slice_count):
        slices_z[z] = slice_positions[z][2]

    # Sort slices by position
    (slices_z, slice_indices) = zip(*sorted(zip(slices_z, slice_indices), reverse=True))

    # Write slices to volume station
    dim = np.array((slice_shape[0], slice_shape[1], slice_count))
    station = np.zeros(dim)

    for z in range(dim[2]):
        slice_z_index = slice_indices[z]
        station[:, :, z] = slices[slice_z_index]

    return station


def stationsFromDicom(input_path_zip, tag):

    # Get slice info
    pixel_data = []
    series = []
    names = []
    positions = []
    pixel_spacings = []
    times = []

    #
    z = zipfile.ZipFile(input_path_zip)

    signal_slice_names = getSignalSliceNamesInZip(z)

    for i in range(len(signal_slice_names)):

        # Read signal slices in memory
        with z.open(signal_slice_names[i]) as f0:

            data = f0.read() # Decompress into memory
            ds = pydicom.read_file(io.BytesIO(data)) # Read from byte stream

            # Little hack for speed
            name = ds.get_item(["0008", "103e"]).value
            if str(tag) in str(name):

                (pixel_data_i, series_i, position_i, spacing_i, time_i) = getDataFromDicom(ds)

                pixel_data.append(pixel_data_i)
                series.append(series_i)
                names.append(name)
                positions.append(position_i)
                pixel_spacings.append(spacing_i)
                times.append(time_i)

    z.close()

    (stat_voxel_data, stat_names, stat_positions, stat_voxel_spacings, stat_times) = groupSlicesToStations(pixel_data, series, names, positions, pixel_spacings, times)

    return (stat_voxel_data, stat_positions, stat_voxel_spacings, stat_times)
