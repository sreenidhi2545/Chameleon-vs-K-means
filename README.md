# Chameleon vs K-Means (Product Clustering)

This project compares two clustering approaches on an e-commerce product dataset:

- `K-Means` (centroid-based clustering)
- `Chameleon-like` clustering using `HDBSCAN` (density-based, used here as an approximation)

The workflow is implemented in a single notebook: `DataMining.ipynb`.

## Project Files

- `DataMining.ipynb`: end-to-end preprocessing, clustering, evaluation, and visualization
- `README.md`: project documentation

## Dataset

The notebook is designed for the Flipkart e-commerce CSV and can load either:

- a direct `.csv` file, or
- a `.zip` containing the CSV

Detected dataset from your archive:

- `flipkart_com-ecommerce_sample.csv` (inside `archive.zip`)

### Required columns

The notebook expects these columns:

- `product_category_tree`
- `retail_price`
- `discounted_price`
- `brand`
- `description`

## Pipeline Summary

1. Load dataset (CSV or ZIP upload in Colab).
2. Select required columns and drop missing values.
3. Build feature matrix with:
   - numeric features: `retail_price`, `discounted_price`
   - one-hot encoded `brand`
   - TF-IDF vectors from `description` (`max_features=300`)
4. Standardize features and reduce dimensions with `PCA(n_components=15)`.
5. Train:
   - `HDBSCAN(min_cluster_size=40)` as Chameleon-like clustering
   - `KMeans(n_clusters=5, random_state=42)`
6. Compare silhouette scores.
7. Visualize clusters with heatmap and 2D scatter plots.

## Requirements

Install dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn hdbscan jupyter
```

If you run in Google Colab, install `hdbscan` first:

```python
!pip install hdbscan
```

## How to Run

### Option 1: Google Colab (current notebook flow)

1. Open `DataMining.ipynb` in Colab.
2. Run all cells.
3. Upload either:
   - `flipkart_com-ecommerce_sample.csv`, or
   - `archive.zip` containing that CSV.

### Option 2: Local Jupyter (small notebook edit)

The current notebook uses `from google.colab import files` and `files.upload()`.

For local Jupyter, replace the upload block with direct file loading, for example:

```python
df = pd.read_csv("data/flipkart_com-ecommerce_sample.csv", encoding="latin1")
```

## Add Dataset to GitHub

Recommended structure:

```text
Chameleon-vs-K-means/
  DataMining.ipynb
  README.md
  data/
    flipkart_com-ecommerce_sample.csv
```

Then commit normally:

```bash
git add .
git commit -m "Add dataset and notebook documentation"
```

Notes:

- Your CSV is about 38 MB, so it is below GitHub's 100 MB single-file limit.
- If the dataset grows large later, use Git LFS.

## Output

The notebook prints final silhouette scores for both algorithms and a conclusion about which clustering method performed better on the processed feature set.
