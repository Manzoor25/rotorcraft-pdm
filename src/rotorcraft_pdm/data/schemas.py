# =============================================================
# WHAT THIS FILE DOES:
# Data contracts. A contract is a written promise about the shape
# of a table. If a download is corrupt, we fail here instead of
# training a model on nonsense.
#
# HOW IT CONNECTS TO OTHER FILES:
# - cmapss.py produces the tables these schemas check
# - tests/test_schemas.py checks both a good table and a bad one
#
# WHY PANDERA:
# It is a small validation library for pandas tables. The alternative
# is a pile of if-statements. The if-statements work. Pandera keeps
# the promise next to the column names.
# =============================================================

import pandera.pandas as pa
from pandera.pandas import Check, Column

from rotorcraft_pdm.data.cmapss import COLUMN_NAMES


# --------------------------------------------------------------
# FUNCTION: cmapss_schema
# WHAT IT DOES: Builds the contract for a raw C-MAPSS table
# INPUTS: None
# OUTPUTS: A Pandera DataFrameSchema
# --------------------------------------------------------------
def cmapss_schema():
    columns = {
        "unit": Column(int, checks=Check.ge(1)),
        "cycle": Column(int, checks=Check.ge(1)),
    }
    for name in COLUMN_NAMES:
        if name in columns:
            continue
        columns[name] = Column(float, nullable=False)
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


# --------------------------------------------------------------
# FUNCTION: validate_cmapss
# WHAT IT DOES: Checks a table against the contract
# INPUTS: frame
# OUTPUTS: The same frame if it passed. Raises if it did not.
# --------------------------------------------------------------
def validate_cmapss(frame):
    return cmapss_schema().validate(frame)
