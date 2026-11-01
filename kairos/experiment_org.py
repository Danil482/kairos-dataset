"""As-of organisation features from other authors in an explicit fit bank.

This module never reads labels or reactions. Call it separately for each outer
training partition: held-out authors may supply their own earlier ORG history,
but never the global popularity or co-occurrence evidence.
"""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd


KEYS = ["author", "post"]
FEATURE_COLUMNS = [
    "org_net_n_orgs", "org_net_known_fraction", "org_net_log_authors_mean",
    "org_net_log_authors_max", "org_net_cooccurrence_fraction",
    "org_net_no_orgs", "org_net_all_unseen", "org_net_no_own_org_history",
]


def organization_features(history: pd.DataFrame, queries: pd.DataFrame,
                          bank_authors: set) -> pd.DataFrame:
    """Return KEYS and eight numeric features, preserving queries' index/order.

    Queries require only author/post; their ORGs and publication times come from
    the uniquely matching history rows. ORG codes are deduplicated per post.
    Only posts with posted_at < query.posted_at are visible, including when
    several authors publish at the same timestamp. Query-owner evidence is
    excluded from the bank even when that owner is a training author.

    Popularity is log1p of the number of distinct *other bank authors* seen per
    current ORG, averaged/maximised across all current ORGs (unknowns are zero).
    Co-occurrence is the fraction of current ORGs connected to any ORG in the
    query owner's own earlier history by a previous bank post from another
    author. Edges require two distinct ORGs in one source post; no self-loops.

    Fractions and popularity are zero for an empty current ORG set. no_orgs
    distinguishes that case from all_unseen, which requires a nonempty set.
    no_own_org_history means no earlier ORG, not necessarily no earlier post.
    The input frames are never modified.
    """
    for name, frame in (("history", history), ("queries", queries)):
        if not set(KEYS) <= set(frame):
            raise ValueError(f"{name} requires author and post columns")
        if frame[KEYS].isna().any().any() or frame[KEYS].eq("").any().any():
            raise ValueError(f"{name} has missing author/post keys")
    if history.duplicated(KEYS).any():
        raise ValueError("history has duplicate author/post keys")
    if not {"posted_at", "org_codes"} <= set(history):
        raise ValueError("history requires posted_at and org_codes")
    dates = pd.to_datetime(history.posted_at, utc=True, errors="raise")
    if dates.isna().any():
        raise ValueError("history has missing publication timestamps")
    codes = []
    for value in history.org_codes:
        if (not isinstance(value, (tuple, list, set, frozenset))
                or any(not isinstance(code, str) or not code.startswith("ORG_")
                       for code in value)):
            raise ValueError("org_codes must contain collections of ORG_ codes")
        codes.append(tuple(sorted(set(value))))

    output = queries[KEYS].copy()
    values = np.zeros((len(queries), len(FEATURE_COLUMNS)), dtype=np.float64)
    lookup = pd.MultiIndex.from_frame(history[KEYS])
    positions = lookup.get_indexer(pd.MultiIndex.from_frame(queries[KEYS]))
    if (positions < 0).any():
        raise ValueError("query author/post keys are absent from history")
    query_positions = {}
    for position, source in enumerate(positions):
        query_positions.setdefault(int(source), []).append(position)

    bank = set(bank_authors)
    owners = history.author.to_numpy()
    query_authors = set(queries.author)
    times = dates.astype("int64").to_numpy()
    order = np.argsort(times, kind="stable")
    last_query_time = times[positions].max() if len(positions) else None
    own_past = {author: set() for author in query_authors}
    org_authors = {}
    # For an edge, one author ID suffices until a second distinct author occurs.
    # None then means >=2 authors: at least one must differ from any query owner.
    cooccurrence = {}

    start = 0
    while start < len(order) and last_query_time is not None:
        timestamp = times[order[start]]
        if timestamp > last_query_time:
            break
        end = start + 1
        while end < len(order) and times[order[end]] == timestamp:
            end += 1
        batch = order[start:end]
        # All queries precede all updates at this timestamp, regardless of input order.
        for source in batch:
            destinations = query_positions.get(int(source))
            if not destinations:
                continue
            owner, current = owners[source], codes[source]
            past = own_past[owner]
            counts, connected = [], 0
            for code in current:
                authors = org_authors.get(code, set())
                counts.append(len(authors) - int(owner in authors))
                neighbors = cooccurrence.get(code, {})
                connected += any(neighbors[other] is None or neighbors[other] != owner
                                 for other in past.intersection(neighbors))
            n_orgs = len(current)
            known = sum(count > 0 for count in counts)
            popularity = [math.log1p(count) for count in counts]
            row = [n_orgs, known / max(1, n_orgs),
                   sum(popularity) / max(1, n_orgs), max(popularity, default=0.0),
                   connected / max(1, n_orgs), int(n_orgs == 0),
                   int(n_orgs > 0 and known == 0), int(not past)]
            values[destinations] = row
        for source in batch:
            owner, current = owners[source], codes[source]
            if owner in query_authors:
                own_past[owner].update(current)
            if owner not in bank:
                continue
            for code in current:
                org_authors.setdefault(code, set()).add(owner)
            for left, right in combinations(current, 2):
                for first, second in ((left, right), (right, left)):
                    neighbors = cooccurrence.setdefault(first, {})
                    if second not in neighbors:
                        neighbors[second] = owner
                    elif neighbors[second] != owner:
                        neighbors[second] = None
        start = end

    for column, column_values in zip(FEATURE_COLUMNS, values.T):
        output[column] = column_values
    return output
