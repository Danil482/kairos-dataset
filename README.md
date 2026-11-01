<div align="center">

# Kairos Dataset

**Did the post do better or worse than its author usually does?**

5,433 LinkedIn authors · 151,351 posts · 89,018 labels · 161,589 media files

[Dataset page](https://kairos-labs.ai/dataset/) ·
[Get the data](#get-the-data) ·
[Baseline](#baseline) ·
[Evaluate](#evaluate-your-model) ·
[Rules of use](#rules-of-use)

![code: MIT](https://img.shields.io/badge/code-MIT-2b6cb0)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-2b6cb0)
![data: on request](https://img.shields.io/badge/data-on%20request-555555)

</div>

**Solution branch:** [reproducible experiments and results](README_SOLUTION.md) ·
[explanatory notebook](solution.ipynb).

This repository is the code half of the Kairos Dataset: it reads the release,
reproduces the label, trains a baseline and scores predictions. The data is not
here. Request it on the [dataset page](https://kairos-labs.ai/dataset/) and unpack
it into `data/`.

## The task

An example first. Author A usually gets about 20 reactions on a post; a post with 80
did four times as well as usual, so it is **above**. Author B usually gets about 300;
the same 80 reactions are a quarter of B's usual, so for B it is **below**. The label
asks how a post did against its own author, not how many reactions it got: a model
fed raw counts learns who wrote the post and never has to read it.

```
level = median of log(1 + reactions) over the author's own earlier posts, at least five
resid = log(1 + reactions of the post) - level
```

| class | `label` | residual | reactions against the author's usual | test share |
|---|:---:|---|---|---:|
| below | 0 | < -0.549 | under ×0.577 | 20.1 % |
| typical | 1 | in between | in between | 59.5 % |
| above | 2 | > 1.157 | over ×3.182 | 20.4 % |

The cutoffs are the 20th and 80th percentiles of the train residuals, applied
unchanged to test. An author's first five own posts carry no label, since they form
the first level, and neither do plain reposts, which arrive with the original post's
counters. `python -m kairos.label` recomputes all 89,018 labels from the posts and
checks them against `labels.csv`.

**Split.** 80 / 20 by author, seed 42: 3,295 train authors with 71,395 labelled posts,
824 test authors with 17,623. All posts of one author sit on one side, because the
label is defined against that author. Train on train, report on test.

**Metrics.** macro-F1 is the headline. It weighs the three classes equally, so
answering *typical* everywhere scores 0.25 instead of the 0.60 that accuracy would
give it. Quadratic weighted kappa (QWK) sits beside it, because the classes are
ordered and calling a *below* post *above* is a worse mistake than calling it
*typical*.

## Get the data

1. **Request the password** on the [dataset page](https://kairos-labs.ai/dataset/). It
   arrives by e-mail.
2. **Download the core archive**, the five volumes `lab1_core.7z.001` to `.005`
   (50.5 GB): text, labels, the media manifest, images and PDFs. The baseline needs
   nothing else. The video archive (11 volumes, 114.1 GB) adds the post videos.
3. **Check** every volume against its SHA-256 on the dataset page:
   ```
   certutil -hashfile lab1_core.7z.001 SHA256    # Windows
   sha256sum lab1_core.7z.*                      # Linux / macOS
   ```
4. **Unpack into `data/`** of this repository. 7-Zip asks for the password and picks
   up the other volumes itself:
   ```
   7z x lab1_core.7z.001 -odata
   ```
   7-Zip on Windows, Keka on macOS, `p7zip-full` on Linux. The video archive unpacks
   into the same `data/media/`.

```
data/
├── authors/AUTHOR_xxxx.json    one profile per author
├── posts/AUTHOR_xxxx.json      the author's feed as it was served; urn is the post code
├── media/<ab>/<sha256>.<ext>   every media file, once, named by its content hash
├── media/manifest.jsonl        which post carries which file
├── labels.csv                  author, post, split, label, class
└── split.json                  train and test authors, and the cutoffs
```

Keep the data somewhere else? Every script takes `--data <folder>`.

## Quickstart

```
git clone https://github.com/Danil482/kairos-dataset.git
cd kairos-dataset
pip install -r requirements.txt

python -m kairos.label     # recompute the labels and check them against labels.csv
python baseline.py         # train and score the baseline, under a minute on a laptop
python evaluate.py --predictions results/predictions/baseline.csv
```

## Baseline

Gradient boosting over everything in the release except the words, trained on the
train authors with balanced class weights and scored on the 17,623 test posts. It
sees what the post is (length, hashtags, mentions, links, type, posting time), who
wrote it (the author's level and spread, how often they post, followers) and when
the post was crawled. It never reads the text and never looks at the media.

| | macro-F1 | QWK | accuracy |
|---|---:|---:|---:|
| **baseline** | **0.5410** | **0.4576** | **0.5465** |
| always *typical* | 0.2488 | 0.0000 | 0.5953 |
| random, in the train class mix | 0.3319 | -0.0080 | 0.4386 |

The bar to beat is **0.5410 macro-F1**. The words and the media are yours.

### For reference: Claude, zero-shot

How far does a frontier model get without any training? In October 2026 Claude Opus 5.5
and Claude Sonnet 5.5 each answered a uniform random sample of **2,000 of the 17,623
test posts** (630 of the 824 test authors). These numbers hold for those 2,000 posts
only: the baseline scores 0.5496 on them, not 0.5410, so read the Claude rows against
that row. Intervals are 95 %, resampling authors.

| on the same 2,000 test posts | macro-F1 | QWK | accuracy |
|---|---:|---:|---:|
| baseline | 0.5496 [0.526, 0.572] | 0.4592 | 0.5575 |
| Claude Sonnet 5.5, zero-shot, with the author's last five posts | 0.5167 [0.491, 0.541] | 0.3787 | 0.6625 |
| Claude Opus 5.5, zero-shot, with the author's last five posts | 0.6272 [0.603, 0.650] | 0.5420 | 0.6870 |

Each post was a fresh `claude -p` call (Claude Code, effort `low`, no tools), so no
post saw another. The prompt gave the author's usual number of reactions, their five
most recent earlier posts (the first 200 characters, the reactions and the first
picture of each), the new post's text (up to 1,000 characters) and up to four of its
pictures, a video by its poster frame, and asked whether the post would get clearly
fewer reactions than usual (under about ×0.58), about the usual, or clearly more (over
about ×3.2). The class is the letter the model chose; nothing was tuned. Sonnet seldom
calls an extreme class (7 % and 9 % of its answers, against about 20 % each in the
label), which is why its accuracy is high and its macro-F1 low.

These runs sent the posts' text and pictures to Anthropic, by the dataset authors'
decision for this reference row. It does not change the [rules of use](#rules-of-use):
your copy of the media stays inside your course and does not go to an external service.

## Evaluate your model

Write one row per test post with its `author`, `post` and predicted `label`
(0, 1, 2) or `class` (below, typical, above):

```
author,post,label
AUTHOR_xxxx,POST_xxxxxx,1
```

```
python evaluate.py --predictions my_predictions.csv
```

It prints macro-F1, QWK, accuracy, F1 by class and the confusion matrix, and refuses
a file that misses a test post. The baseline's predictions,
`results/predictions/baseline.csv`, are in this format.

## Working with the data

```python
from kairos.data import load_labels, load_posts
from kairos.label import author_level

posts = author_level(load_posts())     # own posts, with the author's level and residual
table = load_labels().merge(posts, on=["author", "post"])
train, test = table[table.split == "train"], table[table.split == "test"]
```

Things that are easy to get wrong:

- **Dates.** `postedAt` in a post is LinkedIn's relative display string ("2yr",
  "1mo"). The timestamp is `postedDateTimestamp`, in milliseconds since the epoch;
  `crawledAt` at the top of a feed file is when the feed was collected.
- **Reactions are one snapshot**, taken at crawl time, a median of 710 days after the
  post was published.
- **Reposts.** A post whose `author.code` is not the feed's owner, or that carries a
  `company` block, is someone else's post that the author reposted, with the
  original's counters. `load_posts(own_only=False)` returns the whole feed with an
  `own` flag. A share through LinkedIn's share dialog is the author's own post and is
  labelled.
- **Placeholders.** People, organisations, handles, e-mails, phones and links in the
  text are replaced by `⟨PERSON_n⟩`, `⟨ORG_n⟩`, `⟨AUTHOR_n⟩`, `⟨HANDLE⟩`, `⟨EMAIL⟩`,
  `⟨PHONE⟩` and `⟨URL⟩`. Author and organisation codes are the same across the
  dataset; person codes are numbered within one author's files.
- **Media.** `media/manifest.jsonl` joins files to posts on `author` and `post`, and
  `from_reshared_post` marks files of the reposted original. Videos are not
  transcribed; 1,332 posts carry LinkedIn's own text recognition of an attached
  document under `document.transcript`.
- **Followers** in `authors/*.json` were read in 2026: today's audience, not the one
  that saw the post.

## Rules of use

The text is pseudonymised; the pictures, videos and PDFs are not. Faces, names on
slides, logos and voices in the media are left as they are.

- **Do not re-identify anyone.** Do not try to link a code to a real person, and do
  not publish examples that would let others do it.
- **Keep the data inside your course, project or research.** Do not re-host it or
  pass on the password.

The MIT license of this repository covers the code, not the data.

## Source

Public LinkedIn profiles and feeds, collected and sponsored by
[SOMIN](https://somin.ai) in 2026 for the [ITMO University](https://itmo.ru) course
*Social Media Analytics, Agents & LLMs*.

## Citation

```bibtex
@misc{kairos2026,
  author       = {{Kairos}},
  title        = {Kairos Dataset: post engagement against the author's own level},
  year         = {2026},
  howpublished = {\url{https://kairos-labs.ai/dataset/}},
  note         = {Data sponsored by SOMIN}
}
```

## License

Code: [MIT](LICENSE). Data: the [rules of use](#rules-of-use) above.
