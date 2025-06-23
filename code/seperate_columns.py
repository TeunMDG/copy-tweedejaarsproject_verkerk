import pandas as pd
import os

def split_csv_columns(input_path, output_dir=None, index_col=None, keep_index=False):
    """
    Splits a CSV file into separate files for each column.

    Args:
        input_path (str): Path to input CSV file
        output_dir (str): Custom output directory (default: 'split_{input_filename}')
        index_col (str): Column to use as index (optional)
        keep_index (bool): Whether to include index in output files

    Returns:
        dict: {output_directory: path, generated_files: [list_of_paths]}
    """
    # Read input file
    try:
        df = pd.read_csv(input_path, index_col=index_col)
    except Exception as e:
        raise ValueError(f"Failed to read {input_path}: {str(e)}")

    # Create output directory
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = f"split_{base_name}"

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    # Process each column
    for column in df.columns:
        # Handle filename sanitization
        safe_colname = "".join(c if c.isalnum() else "_" for c in column)
        output_path = os.path.join(output_dir, f"{safe_colname}.csv")

        # Save single-column DataFrame
        if keep_index and df.index.name:
            df[[column]].to_csv(output_path)
        else:
            df[column].to_csv(output_path, header=True)

        generated_files.append(output_path)

    return {
        "output_directory": os.path.abspath(output_dir),
        "generated_files": generated_files,
        "original_columns": df.columns.tolist()
    }