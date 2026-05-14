import pandas as pd
from .config import DATA_PATH


def load_perfume_data(path=DATA_PATH):
    """Load the original perfume Excel dataset."""
    return pd.read_excel(path)
