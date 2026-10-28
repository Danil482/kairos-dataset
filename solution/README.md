# Solution code

| File | Purpose |
|---|---|
| `config.json` | Fixed configuration: four components, their features and hyperparameters, stack settings, pinned E5 and CLIP revisions. |
| `prepare.py` | Builds feature tables in `work/` from `data/`: encodes text with E5 and images with CLIP locally, caches the vectors. |
| `content.py` | Post features: text and attachment fields, author history from strictly earlier posts, shares of earlier posts below and above the cutoffs, similarity to earlier posts, visual history. |
| `models.py` | The four components, the logistic stack over their log-odds, class weights tuned for macro-F1. |
| `train.py` | Loads the feature tables and trains the stack on all train authors. |

Run everything through `improved.py` in the repository root, see the main README.

Inputs: the Kairos core release in `data/` (posts, authors, labels, split, images). Test labels are masked during preparation. Current-post reactions, labels and IDs never enter the features.
