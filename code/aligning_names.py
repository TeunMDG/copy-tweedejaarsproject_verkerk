import pandas as pd
import argparse
import os
import re

def get_ventilator_id(name_string):
    """
    Extracts the unique identifier from a ventilator name string.
    The identifier is assumed to be the text following the word "ventilator".

    Args:
        name_string (str): The full ventilator name (e.g., "Main-unit-ventilator-Alpha-1").

    Returns:
        str: The extracted identifier (e.g., "alpha-1"), in lowercase and stripped of
             leading/trailing hyphens or spaces. Returns the original name in lowercase
             if "ventilator" is not found.
    """
    # Use a case-insensitive search to find "ventilator"
    match = re.search('ventilator', name_string, re.IGNORECASE)
    if match:
        # Get the part of the string after "ventilator"
        identifier = name_string[match.end():]
        # Clean it up by stripping leading/trailing junk and making it lowercase
        return identifier.strip(' -_').lower()
    # Fallback if the pattern isn't found
    return name_string.lower()

def align_ventilator_data(master_order_file, data_file1, data_file2):
    """
    Aligns the columns of two data CSV files based on a master order file,
    using partial name matching to identify columns.

    Args:
        master_order_file (str): Path to the CSV file defining the target ventilator order.
                                 This file should have one column with ventilator names.
        data_file1 (str): Path to the first data CSV file (e.g., energy use).
        data_file2 (str): Path to the second data CSV file (e.g., frequency).
    """
    try:
        # --- Step 1: Read the master order and create a target list of IDs ---
        print(f"Reading master order from '{os.path.basename(master_order_file)}'...")
        master_order_df = pd.read_csv(master_order_file, header=None)
        # Create a list of the target ventilator IDs in the desired order
        target_id_order = [get_ventilator_id(name) for name in master_order_df[0].tolist()]
        print(f"Target ID order established: {target_id_order}\n")

        # --- Step 2: Read the two main data files ---
        print("Reading data files...")
        df1 = pd.read_csv(data_file1)
        df2 = pd.read_csv(data_file2)
        print(f"'{os.path.basename(data_file1)}' original columns: {list(df1.columns)}")
        print(f"'{os.path.basename(data_file2)}' original columns: {list(df2.columns)}\n")

        # --- Step 3: Define a reusable function to reorder dataframes ---
        def reorder_dataframe(df, df_name_str):
            """Reorders a dataframe's columns based on the target_id_order."""
            print(f"Aligning columns for '{df_name_str}'...")
            # Create a mapping from the dataframe's column IDs to its full column names
            df_id_map = {get_ventilator_id(col): col for col in df.columns}

            # --- Validation Step: Check if all required IDs exist in the dataframe ---
            missing_ids = set(target_id_order) - set(df_id_map.keys())
            if missing_ids:
                raise ValueError(f"Error: '{df_name_str}' is missing columns for the following IDs: {missing_ids}")

            # Create the new column order using the full names from the dataframe
            new_column_order = [df_id_map[id] for id in target_id_order]

            # Re-index the dataframe. This rearranges the columns into the correct order.
            df_aligned = df[new_column_order]
            print(f"'{df_name_str}' new column order: {list(df_aligned.columns)}\n")
            return df_aligned

        # --- Step 4: Reorder both dataframes using the helper function ---
        df1_aligned = reorder_dataframe(df1, os.path.basename(data_file1))
        df2_aligned = reorder_dataframe(df2, os.path.basename(data_file2))

        # --- Step 5: Write the aligned dataframes to new CSV files ---
        def get_aligned_filename(filepath):
            directory, filename = os.path.split(filepath)
            name, ext = os.path.splitext(filename)
            return os.path.join(directory, f"{name}_aligned{ext}")

        output_path1 = get_aligned_filename(data_file1)
        output_path2 = get_aligned_filename(data_file2)

        print(f"Writing aligned files...")
        df1_aligned.to_csv(output_path1, index=False)
        print(f"Successfully saved aligned file: {output_path1}")

        df2_aligned.to_csv(output_path2, index=False)
        print(f"Successfully saved aligned file: {output_path2}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except (ValueError, KeyError) as e:
        print(f"An error occurred during alignment: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- Set up command-line argument parsing for ease of use ---
    parser = argparse.ArgumentParser(description="Align two CSV data files based on a master ventilator order file.")
    parser.add_argument('--master_order', required=True, help="Path to the master ventilator order CSV file (target order).")
    parser.add_argument('--data1', required=True, help="Path to the first data CSV file (e.g., energy use).")
    parser.add_argument('--data2', required=True, help="Path to the second data CSV file (e.g., frequency).")

    args = parser.parse_args()

    # --- Run the main function with the provided file paths ---
    align_ventilator_data(args.master_order, args.data1, args.data2)