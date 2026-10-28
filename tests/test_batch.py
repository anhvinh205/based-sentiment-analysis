import pandas as pd
import pytest

from viabsa.batch import select_review_rows


@pytest.mark.parametrize(
    ("values", "limit", "expected", "skipped"),
    [
        (["review", "", "  ", None], 4, ["review"], 3),
        (["review", "", "another review"], 2, ["review"], 1),
        (["", "  "], 2, [], 2),
    ],
)
def test_select_review_rows_skips_blanks(values, limit, expected, skipped):
    df = pd.DataFrame({"review": values})

    selected, skipped_count = select_review_rows(df, "review", limit)

    assert selected["review"].tolist() == expected
    assert skipped_count == skipped
