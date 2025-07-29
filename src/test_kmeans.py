import pandas as pd
from kmeans.kmeans import KMeans  # adjust if your class is elsewhere

# ---------- load dataset ----------
df = pd.read_csv(r"C:\ML from scratch\ml-algos-from-scratch\src\datasets\Country-data.csv")
X = df.drop(columns=["country"], errors="ignore")


kmeans = KMeans(
    k=10,
    n_retries=10,
    weights=None,
    initialization_method="kmeans++"
)

kmeans.learn(X, normalize_features=True)


df["cluster"] = kmeans.assignments.astype(int)


print("Centroids:")
print(kmeans.centroids)

print("\nCluster counts:")
print(df["cluster"].value_counts().sort_index())

print(f"\nTotal quality score: {kmeans.quality:.2f}")


if kmeans.alerts:
    print("\nDiagnostics:")
    for alert in kmeans.alerts:
        print("•", alert)
else:
    print("\nNo diagnostics warnings.")


print("\nCountries by cluster:")
for cid in sorted(df["cluster"].unique()):
    print(f"\nCluster {cid}:")
    countries = df[df["cluster"] == cid]["country"]
    for country in countries:
        print("•", country)

df.to_csv("clustered_life.csv", index=False)
print("\nSaved to clustered_life.csv")


top_k = kmeans.find_top_k_by_silhouette(X, k_values=range(2, 4), normalize_features=True)

print("\nTop 3 vrednosti k po silhouette indeksu:")
for k, score in top_k:
    print(f"k = {k}, silhouette = {score:.4f}")