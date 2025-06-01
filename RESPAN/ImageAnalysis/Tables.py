# -*- coding: utf-8 -*-
"""
Table tools and functions for spine analysis
==========


"""

__author__    = 'Luke Hammond <luke.hammond@osumc.edu>'
__license__   = 'GPL-3.0 License (see LICENSE)'
__copyright__ = 'Copyright © 2024 by Luke Hammond'
__download__  = 'http://www.github.com/lahmmond/RESPAN'



import numpy as np
import pandas as pd


# helper

GB = 1024 ** 3


def merge_spine_measurements(df1, df2, settings, logger):
    try:
        if settings.additional_logging_dev:
            logger.info("Initial DataFrame info:")
            print_df_info_merge(df1, "df1", logger)
            print_df_info_merge(df2, "df2", logger)

        # Ensure 'label' is a column in both DataFrames
        if 'label' not in df1.columns:
            df1 = df1.reset_index()
        if 'label' not in df2.columns:
            df2 = df2.reset_index()

        # Convert 'label' to string in both DataFrames
        df1['label'] = df1['label'].astype(str)
        df2['label'] = df2['label'].astype(str)

        # Convert any numpy arrays to lists
        df1 = df1.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        df2 = df2.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        # Merge the DataFrames on the 'label' column
        merged_df = pd.merge(df1, df2, on='label', how='outer', suffixes=('_1', '_2'))

        if settings.additional_logging_dev:
            logger.info("Merged DataFrame info:")
            print_df_info_merge(merged_df, "merged_df", logger)

        # Combine columns with _1 and _2 suffixes
        for col in merged_df.columns:
            if col.endswith('_1'):
                base_col = col[:-2]
                col_2 = base_col + '_2'
                if col_2 in merged_df.columns:
                    merged_df[base_col] = merged_df[col].combine_first(merged_df[col_2])
                    merged_df = merged_df.drop(columns=[col, col_2])
                else:
                    merged_df = merged_df.rename(columns={col: base_col})
            elif col.endswith('_2') and col[:-2] not in merged_df.columns:
                merged_df = merged_df.rename(columns={col: col[:-2]})

        # Fill NaN values with 0
        merged_df = merged_df.fillna(0)

        # Convert 'label' back to int if possible
        try:
            merged_df['label'] = merged_df['label'].astype(int)
        except ValueError:
            logger.warning("Could not convert 'label' back to int. Keeping as string.")

        if settings.additional_logging_dev:
            logger.info("Final merged DataFrame info:")
            print_df_info_merge(merged_df, "final_merged_df", logger)

        return merged_df

    except Exception as e:
        logger.error(f"An error occurred in merge_spine_measurements: {str(e)}")
        logger.info("Printing additional information about the DataFrames:")
        print_df_info_merge(df1, "df1", logger)
        print_df_info_merge(df2, "df2", logger)
        raise  # Re-raise the exception after printing debug info


def create_spine_summary_neuron(filtered_table, filename, dendrite_length, dendrite_volume, settings):
    #create summary table

    #spine_reduced = filtered_table.drop(columns=['label', 'z', 'y', 'x'])
    updated_table = filtered_table.iloc[:, 4:]
    #drop dendrite_id column
    updated_table = safe_drop_columns(updated_table, ['spine_id', 'x', 'y', 'z', 'spine_type'])

    #spine_summary = updated_table.mean()
    spine_summary = pd.DataFrame([updated_table.mean()])
    #spine_summary = summary.groupby('dendrite_id').mean()
    #spine_counts = summary.groupby('dendrite_id').size()

    spine_summary = spine_summary.add_prefix('avg_')
    #spine_summary.reset_index(inplace=True)
    #spine_summary.index = spine_summary.index + 1
    # update summary with additional metrics
    spine_summary.insert(0, 'Filename', filename)  # Insert a column at the beginning
    spine_summary.insert(1, 'res_XY', settings.input_resXY)
    spine_summary.insert(2, 'res_Z', settings.input_resZ)
    #spine_summary.insert(3, 'dendrite_length', dendrite_length)
    spine_summary.insert(3, 'dendrite_length', dendrite_length * settings.input_resXY)
    #spine_summary.insert(5, 'dendrite_vol', dendrite_volume)
    spine_summary.insert(4, 'dendrite_vol', dendrite_volume * settings.input_resXY*settings.input_resXY*settings.input_resZ)
    spine_summary.insert(5, 'total_spines', filtered_table.shape[0])
    spine_summary.insert(6, 'spines_per_um', spine_summary['total_spines']/spine_summary['dendrite_length'])
    spine_summary.insert(7, 'spines_per_um3', spine_summary['total_spines']/spine_summary['dendrite_vol'])

    return spine_summary


def safe_drop_columns(df, columns_to_drop, axis=1):
    """
    Safely drop columns from a DataFrame without crashing if columns don't exist.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns_to_drop : list or str
        Column name(s) to drop
    axis : int, default 1
        Axis to drop from (1 for columns, 0 for rows)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with specified columns removed (if they existed)
    """

    # Convert single column name to list
    if isinstance(columns_to_drop, str):
        columns_to_drop = [columns_to_drop]

    # Find which columns actually exist
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    # Drop only the columns that exist
    if existing_columns:
        return df.drop(existing_columns, axis=axis)
    else:
        return df




def create_spine_summary_dendrite(filtered_table, filename, dendrite_lengths, dendrite_volumes, settings, locations):
    #create summary table

    #filtered_table = filtered_table.drop(['spine_id', 'x', 'y', 'z','spine_type'], axis=1)
    filtered_table = safe_drop_columns(filtered_table, ['spine_id', 'x', 'y', 'z', 'spine_type'])

    spine_counts = filtered_table.groupby('dendrite_id').size().reset_index(name='total_spines')

    mean_metrics = filtered_table.groupby('dendrite_id').mean().reset_index()

    spine_summary = pd.merge(mean_metrics, spine_counts, on='dendrite_id')

    # update summary with additional metrics
    spine_summary.insert(0, 'Filename', filename)  # Insert a column at the beginning
    spine_summary.insert(1, 'res_XY', settings.input_resXY)
    spine_summary.insert(2, 'res_Z', settings.input_resZ)
   # spine_summary.insert(4, 'total_spines', spine_counts)

    spine_summary['dendrite_length'] = spine_summary['dendrite_id'].map(dendrite_lengths)
    spine_summary['dendrite_volume'] = spine_summary['dendrite_id'].map(dendrite_volumes)


    spine_summary.rename(columns={'avg_dendrite_length': 'dendrite_length'}, inplace=True)
    spine_summary.rename(columns={'avg_dendrite_vol': 'dendrite_vol'}, inplace=True)
    spine_summary['dendrite_length'] = spine_summary['dendrite_length'] * settings.input_resXY
    spine_summary['dendrite_volume'] = spine_summary['dendrite_volume'] *settings.input_resXY*settings.input_resXY*settings.input_resZ
    #spine_summary = move_column(spine_summary, 'dendrite_length_um', 5)
    #spine_summary = move_column(spine_summary, 'dendrite_vol_um3', 7)
    spine_summary.insert(9, 'spines_per_um', spine_summary['total_spines']/spine_summary['dendrite_length'])
    spine_summary.insert(10, 'spines_per_um3', spine_summary['total_spines']/spine_summary['dendrite_volume'])

    desired_order = ['Filename', 'res_XY', 'res_Z', 'dendrite_id', 'dendrite_length', 'dendrite_volume', 'total_spines', 'spines_per_um', 'spines_per_um3'] + \
                    [col for col in spine_summary.columns if col not in ['Filename', 'res_XY', 'res_Z', 'dendrite_id',
                                                                         'dendrite_length', 'dendrite_volume',
                                                                         'total_spines', 'spines_per_um',
                                                                         'spines_per_um3']] + \
                    []
    spine_summary = spine_summary[desired_order]

    spine_summary.to_csv(locations.tables + filename + '_dendrite_summary.csv', index=False)





def print_df_info(df, name):
    print(f"DataFrame {name} info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
    print("\nAny columns with object dtype:")
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        print(f"Column '{col}' unique values:")
        print(df[col].unique())
    print("\n")

def print_df_info_merge(df, name, logger):
    logger.info(f"Info for {name}:")
    logger.info(f"Shape: {df.shape}")
    logger.info("Columns:")
    for col in df.columns:
        logger.info(f"  {col}: {df[col].dtype}")
    logger.info(f"Index: {df.index}")
    logger.info(f"Index type: {type(df.index)}")
    logger.info("First few rows:")
    logger.info(df.head())
    logger.info("\n")


def move_column(df, column, position):
    """
    Move a column in a DataFrame to a specified position.

    Parameters:
    - df: pandas.DataFrame.
    - column: The name of the column to move.
    - position: The new position (index) for the column (0-based).

    Returns:
    - DataFrame with the column moved to the new position.
    """
    cols = list(df.columns)
    cols.insert(position, cols.pop(cols.index(column)))
    return df[cols]


def reorder_spine_table_columns(spine_table):
    """
    Reorder spine table columns into a sensible, explicit order.
    Safely handles missing columns and automatically detects all intensity channels.

    Parameters:
    -----------
    spine_table : pandas.DataFrame or similar
        Input table with spine analysis data

    Returns:
    --------
    spine_table : pandas.DataFrame
        Table with reordered columns, or original table if reordering fails
    """

    try:
        import re

        # Get current columns
        current_columns = list(spine_table.columns)

        # Define the base column order (without intensity channels)
        base_order = [
            # Core identification and location
            'spine_id', 'x', 'y', 'z', 'dendrite_id',

            # Distance measurements
            'geodesic_dist_to_soma', 'euclidean_dist_to_soma',

            # Spine measurements
            'spine_area', 'spine_vol', 'spine_surf_area', 'spine_length',
            'spine_length_euclidean', 'spine_bbox_vol', 'spine_extent',
            'spine_solidity', 'spine_convex_vol',

            # Head measurements
            'head_area', 'head_vol', 'head_surf_area', 'head_length',
            'head_euclidean_dist_to_dend', 'head_bbox_vol', 'head_extent',
            'head_solidity', 'head_convex_vol', 'head_convex_hull_ratio',

            # Neck measurements
            'neck_area', 'neck_vol', 'neck_surf_area', 'neck_length',
            'neck_width_mean', 'neck_width_min', 'neck_width_max',
            'neck_bbox_vol', 'neck_extent', 'neck_solidity', 'neck_convex_vol',

            # C1 intensity measurements
            'spine_C1_mean_int', 'spine_C1_max_int', 'spine_C1_int_density',
            'head_C1_mean_int', 'head_C1_max_int', 'head_C1_int_density',
            'neck_C1_mean_int', 'neck_C1_max_int', 'neck_C1_int_density',

            # Classification (will be placed at end)
            'spine_type'
        ]

        # Detect all intensity channels (C2, C3, C4, etc.)
        intensity_pattern = r'^(spine|head|neck)_C(\d+)_(mean|max)_int$'
        intensity_cols = []

        for col in current_columns:
            match = re.match(intensity_pattern, col)
            if match:
                region, channel_num_str, stat_type = match.groups()
                channel_num = int(channel_num_str)

                # Skip C1 channels as they're handled explicitly in base_order
                if channel_num != 1:
                    intensity_cols.append((channel_num, region, stat_type, col))

        # Sort intensity columns by channel number, then region, then stat type
        region_order = {'spine': 0, 'head': 1, 'neck': 2}
        stat_order = {'mean': 0, 'max': 1, 'int_density': 2}
        intensity_cols.sort(key=lambda x: (x[0], region_order[x[1]], stat_order[x[2]]))
        intensity_column_names = [col[3] for col in intensity_cols]

        # Insert intensity columns after C1 columns, before spine_type
        if 'spine_type' in base_order:
            spine_type_idx = base_order.index('spine_type')
            final_order = base_order[:spine_type_idx] + intensity_column_names + ['spine_type']
        else:
            final_order = base_order + intensity_column_names

        # Filter to only include columns that actually exist
        available_columns = [col for col in final_order if col in current_columns]

        # Add any remaining columns not in our defined order
        remaining_columns = [col for col in current_columns if col not in final_order]
        final_column_order = available_columns + remaining_columns

        # Reorder the dataframe
        spine_table = spine_table[final_column_order]

        return spine_table

    except Exception as e:
        print(f"Warning: Column reordering failed with error: {e}")
        print("Returning table with original column order.")
        return spine_table