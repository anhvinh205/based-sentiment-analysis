# Data statement

## Source and licence

`TODO (required before publishing):` where the data comes from, who collected and labelled it, the
licence/terms of use, and whether redistribution is allowed. **Do not commit the raw CSVs or any review
text to a public repository until this is settled.** `data/` and `reports/errors_*.csv` are git-ignored.

## Content

Vietnamese-language lipstick product reviews with one label per aspect category.

| | Raw rows | After cleaning |
|---|---:|---:|
| train | 12,981 | 12,428 |
| validation | 1,623 | 1,545 |
| test | 1,623 | 1,534 |

* **Aspects (8):** stayingpower, texture, smell, price, others, colour, shipping, packing.
* **Labels:** positive / negative / neutral, or empty = aspect not mentioned (encoded as class `none`).
* `others` occurs only as *neutral*. `price` and `packing` have very few negative or neutral reviews
  (15 / 22 and 85 / 15 in the raw training split).
* `full_data.csv` in the original archive is exactly train + val + test; it is **not** used, because it
  contains the test reviews.

## Known issues handled by `viabsa.data.prepare`

* About one in five reviews contains line breaks inside a quoted CSV cell → read with a CSV parser.
* ~2% of training rows have labels but no usable text after cleaning (only URLs or keyboard mash) → dropped.
* Exact and punctuation-insensitive duplicates are removed; texts with conflicting labels are dropped
  (7 in train, 3 in val); reviews in val/test that also occur in train (or test in val) are removed
  (20 and 31 rows). All counts are written to `reports/data_stats.json`.
* ~7% of reviews follow a marketplace auto-form ("Công dụng: … Kết cấu: …"); ~8% contain keyboard-mash
  tokens. Both are flagged (`is_template`, `had_gibberish`) for slice analysis.

## Caveats

Reviews are informal, contain abbreviations, missing diacritics and code-mixing. Labels may be noisy;
inter-annotator agreement is unknown. Reviews are user-generated text and may contain personal
information; do not publish them.
