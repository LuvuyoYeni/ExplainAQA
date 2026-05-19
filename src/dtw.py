import json
import numpy as np


def dtw(ref, query, band_ratio=0.1):
    n, m = len(ref), len(query)
    w = max(int(band_ratio * max(n, m)), abs(n - m))

    D = np.full((n, m), np.inf)

    for i in range(n):
        for j in range(max(0, i - w), min(m, i + w + 1)):
            d = np.sqrt(np.sum((ref[i] - query[j]) ** 2))
            if i == 0 and j == 0:
                D[i, j] = d
            else:
                prev = np.inf
                if i > 0:           prev = min(prev, D[i-1, j])
                if j > 0:           prev = min(prev, D[i, j-1])
                if i > 0 and j > 0: prev = min(prev, D[i-1, j-1])
                if prev < np.inf:
                    D[i, j] = d + prev

    path = []
    i, j = n - 1, m - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            _, i, j = min(
                (D[i-1, j-1], i-1, j-1),
                (D[i-1, j],   i-1, j),
                (D[i,   j-1], i,   j-1),
            )
        path.append((i, j))
    path.reverse()

    return path, float(D[n-1, m-1])


def warp_to_reference(query, path):
    ref_len = path[-1][0] + 1
    buckets = [[] for _ in range(ref_len)]
    for i, j in path:
        buckets[i].append(j)
    return np.array([np.mean(query[idxs], axis=0) for idxs in buckets])


def main():
    with open("data/sample_reps.json") as f:
        reps = json.load(f)

    reps_np = [np.array(r, dtype=np.float64) for r in reps]
    ref = reps_np[0]
    print(f"Reference rep (rep 0): {len(ref)} frames")

    aligned = [ref.tolist()]

    for idx, q in enumerate(reps_np[1:], start=1):
        path, cost = dtw(ref, q, band_ratio=0.1)
        warped = warp_to_reference(q, path)
        print(f"Rep {idx}: {len(q)} frames -> {len(warped)} frames, cost={cost:.2f}")
        aligned.append(warped.tolist())

    with open("data/aligned_reps.json", "w") as f:
        json.dump(aligned, f)
    print("Saved to data/aligned_reps.json")


if __name__ == "__main__":
    main()
