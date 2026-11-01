"""A small adaptation of time-aware author history (Lecture 2, slide 49)."""

import numpy as np
import pandas as pd

RECENT_COLUMNS = [
    "recent3_delta",
    "recent10_delta",
    "last_post_delta",
    "recent3_spread",
]


def recent_author_features(posts):
    table = posts[["author", "post", "posted_at", "reactions"]].reset_index(drop=True)
    if table.duplicated(["author", "post"]).any():
        raise ValueError("Duplicate author/post keys")
    logs = np.log1p(table.reactions.clip(lower=0).to_numpy(dtype=float))
    result = np.zeros((len(table), 4))
    for _, author in table.groupby("author", sort=False):
        previous = []
        for _, batch in author.sort_values(["posted_at", "post"]).groupby(
            "posted_at", sort=True
        ):
            if previous:
                history = logs[previous]
                usual = np.median(history)
                values = [
                    history[-3:].mean() - usual,
                    history[-10:].mean() - usual,
                    history[-1] - usual,
                    history[-3:].std(),
                ]
                result[batch.index] = values
            previous.extend(batch.index.tolist())
    return pd.concat(
        [table[["author", "post"]], pd.DataFrame(result, columns=RECENT_COLUMNS)],
        axis=1,
    )
