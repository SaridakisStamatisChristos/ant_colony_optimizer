import csv
import sys


xs, ys = [], []
with open(sys.argv[1], "r") as f:
    r = csv.DictReader(f)
    for row in r:
        xs.append(row["model"])
        ys.append(float(row["sharpe"]))

print("\nSharpe by model:")
for m, s in zip(xs, ys):
    print(f"{m:>18s}: {s:.3f}")
