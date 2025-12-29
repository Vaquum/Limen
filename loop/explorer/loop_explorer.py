import random
import os
import sys
import subprocess
import time

from IPython.display import Javascript, display
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loop.experiment import UniversalExperimentLoop


def loop_explorer(uel: "UniversalExperimentLoop", host: str = "37.27.112.167") -> None:
    """
    Create a Streamlit explorer for datasets produced by the Universal Experiment Loop.

    Args:
        uel (UniversalExperimentLoop): The experiment loop object providing datasets
        host (str): Hostname or IP address for the Streamlit server

    Returns:
        None: None
    """

    port = random.randint(5001, 5500)

    tmp_parquet = "/tmp/historical_data.parquet"

    datasets = {}

    datasets["historical_data"] = uel.data
    datasets["experiment_log"] = uel.experiment_log
    datasets["experiment_confusion_metrics"] = uel.experiment_confusion_metrics
    datasets["experiment_backtest_results"] = uel.experiment_backtest_results

    for key in datasets.keys():
        # Align filenames with streamlit_app.py expectations
        fname = key
        if key == "experiment_confusion_metrics":
            fname = "confusion_metrics"
        elif key == "experiment_backtest_results":
            fname = "backtest_results"

        try:
            datasets[key].to_pandas().to_parquet(f"/tmp/{fname}.parquet")

        except AttributeError:
            datasets[key].to_parquet(f"/tmp/{fname}.parquet")

    # Resolve script path relative to this file to avoid CWD dependence
    script_path = str((Path(__file__).parent / "streamlit_app.py").resolve())
    workdir = str(Path(__file__).parent.resolve())

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        script_path,
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--",
        "--data",
        tmp_parquet,
    ]

    env = os.environ.copy()
    # Ensure the launched Streamlit process can import the editable 'loop' package
    # by explicitly injecting the project root into PYTHONPATH. This avoids
    # environment/path discrepancies when subprocess resolves the interpreter.
    project_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}:{existing_pythonpath}" if existing_pythonpath else project_root
    )
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    started = False
    t0 = time.time()

    while True:
        line = proc.stdout.readline()

        if not line:
            if proc.poll() is not None:
                raise RuntimeError(
                    "Streamlit exited before starting. Check logs above."
                )
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
    print(f"Open:{url}")

    return {"url": host, "port": port}
