from pathlib import Path
import json, platform, sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import matplotlib
import pyarrow
import requests

root = Path(__file__).resolve().parents[1]
out = root / "results" / "diagnostics" / "environment_check.json"
out.parent.mkdir(parents=True, exist_ok=True)

probe = pd.DataFrame({"pais": ["ECU", "MEX", "FIN"], "valor": [1.0, 2.0, 3.0]})
probe_path = root / "data" / "interim" / "prueba_escritura.parquet"
probe.to_parquet(probe_path, index=False)
read_back = pd.read_parquet(probe_path)
assert probe.equals(read_back)

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "executable": sys.executable,
    "packages": {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "pyarrow": pyarrow.__version__,
        "requests": requests.__version__,
    },
    "project_root": str(root),
    "parquet_read_write": True,
}
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nOK: diagnostico guardado en {out}")
