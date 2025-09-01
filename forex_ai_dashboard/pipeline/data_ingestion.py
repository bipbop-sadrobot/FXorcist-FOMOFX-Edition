import pandas as pd

def load_data(file_path="data/data/raw/ejtrader_eurusd_m1.csv"):
    """
    Loads Forex data from a CSV file and returns a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with 'date' and 'target' columns.
    """
    try:
        # Load the data from the CSV file
        df = pd.read_csv(file_path)

        # Ensure the 'date' column exists
        if 'date' not in df.columns:
            raise ValueError("The CSV file must contain a 'date' column.")

        # Ensure the 'close' column exists
        if 'close' not in df.columns:
            raise ValueError("The CSV file must contain a 'close' column.")

        # Rename 'close' to 'target'
        df = df.rename(columns={'close': 'target'})

        # Convert the 'date' column to datetime objects
        df['date'] = pd.to_datetime(df['date'])

        # Set the 'date' column as the index
        df = df.set_index('date')

        # Sort the DataFrame by date
        df = df.sort_index()

        # Select only the 'target' column
        df = df[['target']]

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except ValueError as e:
        raise ValueError(str(e))
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")
