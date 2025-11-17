import kagglehub
import pandas as pd

class KaggleLoader:
    """
    A simple dataloader to fetch datasets from Kaggle.
    """
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def df(self) -> pd.DataFrame:

        from tqdm.notebook import tqdm
        # Download latest version
        # path = kagglehub.dataset_download(self.dataset_name)

        path = self.dataset_name

        print("Path to dataset files:", path)

        import pandas as pd
        from pathlib import Path
        path_new = Path(path)
        df = pd.read_pickle(path_new)

        print("Shape:", df.shape)       # rows, columns
        print("Data types:\n", df.dtypes)

        return df

