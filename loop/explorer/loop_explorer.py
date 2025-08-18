import random
import os, sys, subprocess, time
from IPython.display import Javascript, display
from pathlib import Path


def loop_explorer(data, host='37.27.112.167'):

    '''
    Visualize the data using Streamlit.

    Args:
        data (pd.DataFrame): The data to visualize.
        host (str): The host to run the Streamlit server on.

    Returns:
        None
    '''

    port = random.randint(5001, 5500)
    
    tmp_parquet = '/tmp/historical_data.parquet'
    
    try:
        data.to_pandas().to_parquet(tmp_parquet)
    
    except AttributeError:
        data.to_parquet(tmp_parquet)

    script_path="loop/explorer/streamlit_app.py"
    script_path = str(Path(script_path).resolve())
    workdir = str(Path(script_path).parent.resolve())

    cmd = [
        sys.executable, "-m", "streamlit", "run", script_path,
        "--server.address", "0.0.0.0",
        "--server.port", str(port),
        "--server.headless", "true",
        "--", "--data", tmp_parquet,
    ]

    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd, cwd=workdir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    started = False
    t0 = time.time()
    
    while True:
        
        line = proc.stdout.readline()
        
        if not line:
            if proc.poll() is not None:
                raise RuntimeError("Streamlit exited before starting. Check logs above.")
            if time.time() - t0 > 20:
                raise TimeoutError("Streamlit did not start within 20s.")
            continue
        
        print(line, end="")
        
        if "You can now view your Streamlit app in your browser." in line:
            started = True
            break
        if "Traceback (most recent call last)" in line:
            pass

    if not started:
        raise RuntimeError("Streamlit failed to announce startup.")

    url = f"http://{host}:{port}"
    display(Javascript(f"window.open('{url}', '_blank');"))
    print("Open:", url)
