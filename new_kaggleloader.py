import kagglehub
import pandas as pd
import os
import requests
from tqdm import tqdm


class KaggleLoader:
    """
    Loads CSV + downloads images for MovieGenre dataset.
    Returns a DataFrame with 'local_path' column.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def df(self) -> pd.DataFrame:

        # ---- 1. Download dataset ----
        base_path = kagglehub.dataset_download(self.dataset_name)
        print("Dataset path:", base_path)

        # ---- 2. Load CSV ----
        csv_path = os.path.join(base_path, "MovieGenre.csv")
        df = pd.read_csv(csv_path, encoding="latin1")

        # ---- 3. Local image folder ----
        img_dir = os.path.join(base_path, "PostersLocal")
        os.makedirs(img_dir, exist_ok=True)

        # ---- Helper: filename for each poster ----
        def poster_path(imdbId):
            return os.path.join(img_dir, f"{imdbId}.jpg")

        # ---- 4. Download images if missing ----
        # print("Downloading posters (skipping existing files)...")
        # for idx, row in tqdm(df.iterrows(), total=len(df)):
        #     url = str(row["Poster"])
        #     imdbId = row["imdbId"]
        #     out_path = poster_path(imdbId)

        #     # skip if exists
        #     if os.path.exists(out_path):
        #         continue

        #     if url.startswith("http"):
        #         try:
        #             r = requests.get(url, timeout=10)
        #             if r.status_code == 200:
        #                 with open(out_path, "wb") as f:
        #                     f.write(r.content)
        #             else:
        #                 print(f"HTTP {r.status_code} for {imdbId}")
        #         except Exception as e:
        #             print(f"Failed {imdbId}: {e}")

        # ---- 5. Add local_path column ----
        from collections import Counter

        print("before filtering missing posters:", df.shape)
        df["local_path"] = df["imdbId"].apply(lambda x: poster_path(x))

        df = df[df["local_path"].apply(os.path.exists)].reset_index(drop=True)
        print("After filtering missing posters:", df.shape)
        print(df.dtypes)

        # 1) Drop rows with missing Genre
        df = df[df["Genre"].notna()].reset_index(drop=True)

        # 2) Count all genres (split by "|")
        genre_counter = Counter()
        for g in df["Genre"]:
            genres = [x.strip() for x in g.split("|") if x.strip()]
            genre_counter.update(genres)

        # 3) Take top-10 genres
        top10_genres = {g for g, _ in genre_counter.most_common(5)}
        print("Top 10 genres:", top10_genres)

        # 4) Keep only rows that have at least one top-10 genre
        def has_top_genre(g):
            genres = [x.strip() for x in g.split("|") if x.strip()]
            return any(gen in top10_genres for gen in genres)

        df = df[df["Genre"].apply(has_top_genre)].reset_index(drop=True)
        print("After keeping only movies with at least one top-10 genre:", df.shape)

        # 5) (Important) Remove non-top-10 genres from each row
        def keep_only_top10(g):
            genres = [x.strip() for x in g.split("|") if x.strip()]
            filtered = [gen for gen in genres if gen in top10_genres]
            return "|".join(filtered)

        df["Genre"] = df["Genre"].apply(keep_only_top10)

        print("Example genres after filtering:")
        print(df["Genre"].head())

        return df

if __name__ == "__main__":
    loader = KaggleLoader("neha1703/movie-genre-from-its-poster")
    df = loader.df()
    print(df.head()) 
    
