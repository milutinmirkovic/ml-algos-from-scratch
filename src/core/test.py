import pandas as pd
from features import extract_features_from_dataframe

# =========================
# 1) Dataset od 20 redova
# =========================
data = {
    "sunčano": [True, False, True, True, False, True, False, False, True, True,
                False, True, False, False, True, True, False, True, False, True],
    "boja": ["plava", "crvena", "zelena", "plava", "zelena", "crvena", "plava", "zelena", "crvena", "plava",
             "crvena", "plava", "plava", "zelena", "crvena", "zelena", "crvena", "plava", "zelena", "crvena"],
    "visina": [170, 165, 180, 175, 160, 190, 158, 170, 168, 182,
               162, 177, 159, 166, 185, 172, 164, 178, 169, 174],
    "labela": ["da", "ne", "da", "ne", "ne", "da", "ne", "da", "ne", "da",
               "ne", "da", "ne", "da", "da", "ne", "ne", "da", "ne", "da"]
}

df = pd.DataFrame(data)
X = df.drop(columns=["labela"])
y = df["labela"].tolist()

# =========================
# 2) Učenje Feature objekata
# =========================
features = extract_features_from_dataframe(X, y)

# =========================
# 3) Test verovatnoća
# =========================
for feature in features:
    print(f"\nFeature: {feature.name}  (continuous: {feature.is_continuous})")

    # izaberi smislen x za test
    if feature.is_continuous:
        test_x = int(X[feature.name].median())          # npr. median visina
    else:
        test_x = X[feature.name].iloc[0]                # prva viđena vrednost (plava / True …)

    for cls in sorted(set(y)):
        p = feature.probability(test_x, cls)
        print(f"  P(x={test_x!r} | y={cls}) = {p:.6f}")
