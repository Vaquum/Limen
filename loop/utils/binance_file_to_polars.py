import requests
import zipfile
import io
import polars as pl


def binance_file_to_polars(url: str, has_header: bool) -> pl.DataFrame:

    '''Downloads a Binance datafile to a polars DataFrame.
    
    Args:
        url (str): The url referring to the datafile.
        has_header (bool): Whether the file has a header.

    Returns: 

        pl.DataFrame
    
    '''

    response = requests.get(url)
    response.raise_for_status()

    zip_buf = io.BytesIO(response.content)
    z = zipfile.ZipFile(zip_buf)
    
    csv_filename = next(name for name in z.namelist() if name.lower().endswith(".csv"))
    
    with z.open(csv_filename) as csv_file:
        df = pl.read_csv(csv_file, has_header=has_header)
        
    return df
