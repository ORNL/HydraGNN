
from typing import Any

import numpy as np


def parse_complex_voltage(value: Any) -> complex:
    """Parse OPFLearn complex-voltage values into a Python complex number.

    The OPFLearn CSV stores voltages as strings such as
    "1.0887 - 0.00059j" and may include spaces or Julia-style suffixes.

    Parameters
    ----------
    value:
        Raw field value from CSV/Parquet.

    Returns
    -------
    complex
        Parsed complex value. Missing values are mapped to NaN+NaNj.

    Raises
    ------
    ValueError
        If the value cannot be parsed.
    """
    if value is None:
        return complex(np.nan, np.nan)

    if isinstance(value, complex):
        return value

    if isinstance(value, (float, int, np.number)):
        if np.isnan(value):
            return complex(np.nan, np.nan)
        return complex(value)

    text = str(value).strip()
    if not text:
        return complex(np.nan, np.nan)

    text = text.replace(" ", "")
    text = text.replace("im", "j")

    if text.endswith("i"):
        text = text[:-1] + "j"

    try:
        return complex(text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse complex bus voltage: {value!r}") from exc
