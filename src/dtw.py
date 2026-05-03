import json
import numpy as np


def dtw(reference, query, band_ratio=0.1):
    """
    DTW alignment with Sakoe-Chiba band constraint.

    reference, query : np.ndarray of shape (T, D)
    band_ratio       : band width as a fraction of the longer sequence length.
                       The band is widened to abs(n-m) when sequences differ in
                       length so a valid (0,0)->(n-1,m-1) path always exists.

    Returns
    -------
    path : list of (i, j) index pairs — the optimal warping path
    cost : float — total accumulated DTW cost at cell (n-1, m-1)
    """
    n, m = len(reference), len(query)
    w = max(int(band_ratio * max(n, m)), abs(n - m))

    D = np.full((n, m), np.inf)

    for i in range(n):
        j_lo = max(0, i - w)
        j_hi = min(m - 1, i + w)
        for j in range(j_lo, j_hi + 1):
            dist = np.linalg.norm(reference[i] - query[j])
            if i == 0 and j == 0:
                D[i, j] = dist
            else:
                prev = np.inf
                if i > 0:
                    prev = min(prev, D[i - 1, j])
                if j > 0:
                    prev = min(prev, D[i, j - 1])
                if i > 0 and j > 0:
                    prev = min(prev, D[i - 1, j - 1])
                if prev < np.inf:
                    D[i, j] = dist + prev

    # Backtrack from (n-1, m-1) to (0, 0)
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
                (D[i - 1, j - 1], i - 1, j - 1),
                (D[i - 1, j],     i - 1, j    ),
                (D[i,     j - 1], i,     j - 1),
            )
        path.append((i, j))
    path.reverse()

    return path, float(D[n - 1, m - 1])


def warp_to_reference(query, path):
    """
    Warp query to the reference length using the DTW warping path.

    For each reference index i, collects all query indices j that were mapped
    to it and averages them — producing one output frame per reference frame.

    query : np.ndarray (T_query, D)
    path  : list of (i, j) pairs from dtw()

    Returns np.ndarray of shape (T_reference, D)
    """
    ref_len = path[-1][0] + 1
    buckets = [[] for _ in range(ref_len)]
    for i, j in path:
        buckets[i].append(j)
    return np.array([np.mean(query[idxs], axis=0) for idxs in buckets])


def main():
    with open("data/sample_reps.json") as f:
        reps = json.load(f)

    reps_np = [np.array(rep, dtype=np.float64) for rep in reps]
    reference = reps_np[0]
    ref_len = len(reference)

    print(f"Reference rep (rep 0): {ref_len} frames")
    print("-" * 50)

    aligned = [reference.tolist()]  # rep 0 needs no alignment

    for idx, query in enumerate(reps_np[1:], start=1):
        path, cost = dtw(reference, query, band_ratio=0.1)
        warped = warp_to_reference(query, path)

        print(
            f"Rep {idx}:  original={len(query):>3} frames  |  "
            f"DTW cost={cost:>10.2f}  |  aligned={len(warped):>3} frames"
        )
        aligned.append(warped.tolist())

    print("-" * 50)
    print(f"Aligned lengths: {[len(r) for r in aligned]}")

    with open("data/aligned_reps.json", "w") as f:
        json.dump(aligned, f)
    print("Saved to data/aligned_reps.json")


if __name__ == "__main__":
    main()
