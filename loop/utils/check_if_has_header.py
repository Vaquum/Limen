import fsspec, zipfile, io


def check_if_has_header(url: str, encoding: str = "utf-8") -> bool:
    
    '''
    Compute whether a CSV file contains a header row.
    
    NOTE: This is mainly intended to be used with Binance Market Data files.
    
    Args:
        url (str): URL of the CSV file to check
        encoding (str): Text encoding of the CSV file
        
    Returns:
        bool: True if the CSV file has a header, False otherwise
    '''

    with fsspec.open(url, "rb", block_size=2**20) as remote:
        
        with zipfile.ZipFile(remote) as zf:
            name = zf.namelist()[0]
            
            with zf.open(name) as member:
                line = io.BufferedReader(member).readline(256_000)
                line = line.decode(encoding, "replace").rstrip()
                
                try: 
                    int(line.split(',')[0])
                    has_header = False
                
                except ValueError:
                    has_header = True
            
                return has_header
            