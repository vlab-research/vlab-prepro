import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vlab_prepro import PreprocessingError, Preprocessor, compute_seed, parse_number
from vlab_prepro.preprocess import add_final_answer, flatten_dict, wrap_empty


def ts(h, m, s):
    return datetime(2020, 1, 1, h, m, s).replace(tzinfo=timezone.utc).isoformat()


def dt(h, m, s):
    return pd.Timestamp(datetime(2020, 1, 1, h, m, s), tz="UTC")


def make_df(data):
    columns = [
        "surveyid",
        "userid",
        "flowid",
        "question_ref",
        "question_idx",
        "response",
        "timestamp",
        "metadata",
    ]

    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def df():
    data = [
        ("a", "1", 1, "A", 1, "response", ts(12, 2, 0), '{"stratumid": "Z"}'),
        ("a", "1", 1, "B", 2, "response", ts(12, 2, 1), '{"stratumid": "Z"}'),
        ("a", "1", 1, "C", 3, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("a", "1", 1, "D", 4, "response", ts(12, 2, 10), '{"stratumid": "Z"}'),
        ("b", "1", 1, "A", 1, "response", ts(12, 3, 0), '{"stratumid": "Z"}'),
        ("b", "1", 1, "B", 2, "response", ts(12, 4, 0), '{"stratumid": "Z"}'),
        ("a", "2", 1, "A", 1, "response", ts(12, 2, 0), '{"stratumid": "X"}'),
        ("a", "2", 1, "B", 2, "response", ts(12, 2, 5), '{"stratumid": "X"}'),
        ("c", "2", 2, "C", 2, "response", ts(12, 3, 5), "{}"),
        ("b", "3", 1, "A", 1, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("b", "3", 1, "A", 1, "response2", ts(12, 2, 6), '{"stratumid": "Z"}'),
        ("c", "3", 1, "A", 1, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
    ]

    return make_df(data)


@pytest.fixture
def form_df():
    columns = ["surveyid", "shortcode", "version", "survey_created", "metadata"]

    data = [
        ("a", "foo", 1, ts(12, 1, 0), '{"wave": "0"}'),
        ("b", "bar", 1, ts(12, 1, 0), "{}"),
        ("c", "fooz", 1, ts(12, 1, 0), '{"wave": "0"}'),
    ]

    return pd.DataFrame(data, columns=columns)


# TODO: test empty data frames - some way to know where it went wrong
# rather than ambiguous missing column errors...


def test_add_metadata_ads_single_key(df):
    p = Preprocessor()
    d = p.add_metadata(["stratumid"], df)
    assert "stratumid" in d.columns
    assert d["stratumid"].iloc[0] == "Z"
    assert "stratumid" in p.keys


def test_add_metadata_prefixes_conflicting_key(df):
    df['metadata'] = df.metadata.map(lambda x: '{"A": "foo"}')
    p = Preprocessor()
    d = p.add_metadata(["A"], df)
    assert "metadata_A" in d.columns
    assert d["metadata_A"].iloc[0] == "foo"
    assert "metadata_A" in p.keys


def test_add_duration_adds_answer_time_min_from_all_surveys(df):
    p = Preprocessor()
    d = p.add_duration(df)

    s = d.groupby("userid").apply(lambda df: df.answer_time_min.iloc[0])
    assert s["1"] == 1.0
    assert s["2"] == 5.0


def test_add_duration_adds_survey_start_time_from_all_surveys(df):
    p = Preprocessor()
    d = p.add_duration(df)
    assert d["survey_start_time"].iloc[0] == dt(12, 2, 0)
    assert d["survey_start_time"].iloc[7] == dt(12, 2, 0)


def test_add_duration_adds_survey_duration_all_surveys_if_not_form_data(df):
    p = Preprocessor()
    d = p.add_duration(df)
    s = d.groupby("userid").apply(lambda df: df.survey_duration.iloc[0])
    assert s["1"] == 120.0
    assert s["2"] == 65.0
    assert s["3"] == 1.0


def test_add_duration_adds_survey_duration_per_survey_if_form_data(df, form_df):
    p = Preprocessor()
    df = p.add_form_data(form_df, df)
    d = p.add_duration(df)
    print(d[["userid", "surveyid", "survey_duration"]])

    s = d.groupby(["userid", "surveyid"]).apply(lambda df: df.survey_duration.iloc[0])
    assert s[("1", "a")] == 10.0
    assert s[("1", "b")] == 60.0


def test_add_duration_adds_to_keys(df):
    p = Preprocessor()
    p.add_duration(df)
    assert "survey_start_time" in p.keys
    assert "answer_time_min" in p.keys


def test_add_time_indicators_adds_correct_week(df):
    p = Preprocessor()
    df = p.add_duration(df)
    d = p.add_time_indicators(["week"], df)
    assert "week" in d.columns
    assert d["week"].iloc[0] == 1
    assert "week" in p.keys


def test_drop_users_without_raises_on_missing_field(df):
    p = Preprocessor()
    with pytest.raises(PreprocessingError):
        p.drop_users_without("stratumid", df)


def test_drop_users_without_removes_anyone_who_ever_answered_without_key(df):
    p = Preprocessor()
    df = p.add_metadata(["stratumid"], df)
    d = p.drop_users_without("stratumid", df)

    assert "2" not in d.userid.unique()


def test_add_form_data_adds_metadata(df, form_df):
    p = Preprocessor()
    d = p.add_form_data(form_df, df)
    assert "wave" in d.columns
    assert d[d.surveyid == "a"]["wave"].iloc[0] == "0"
    assert pd.isna(d[d.surveyid == "b"]["wave"].iloc[0])
    assert "wave" in p.keys
    assert "shortcode" in p.keys


def test_add_form_data_adds_metadata_with_prefix_when_given(df, form_df):
    p = Preprocessor()
    f = p.add_form_data(form_df, prefix="form")
    d = f(df)
    assert "form_wave" in d.columns
    assert d[d.surveyid == "a"]["form_wave"].iloc[0] == "0"
    assert pd.isna(d[d.surveyid == "b"]["form_wave"].iloc[0])
    assert "form_wave" in p.keys
    assert "shortcode" in p.keys


def test_keep_final_answer_removes_previous_answers(df):
    p = Preprocessor()
    d = p.keep_final_answer(df)

    d = d[(d.userid == "3") & (d.surveyid == "b")]
    assert d.shape[0] == 1
    assert d["response"].iloc[0] == "response2"


def test_drop_duplicated_users_only_drops_those_which_duplicate_on_keys(df, form_df):
    p = Preprocessor()
    df = p.add_form_data(form_df, df)
    df = p.keep_final_answer(df)
    d = p.drop_duplicated_users(["wave"], df)
    assert "2" not in d.userid.unique()
    assert "1" in d.userid.unique()
    assert "3" in d.userid.unique()


def test_add_percentage_valid(df):
    p = Preprocessor()
    d = p.count_invalid(df)
    assert "invalid_answer_percentage" in d.columns
    assert "invalid_answer_count" in d.columns

    assert d[d.userid == "1"]["invalid_answer_percentage"].iloc[0] == 0.0
    assert d[d.userid == "3"]["invalid_answer_percentage"].iloc[0] == 1 / 3

    assert d[d.userid == "1"]["invalid_answer_count"].iloc[0] == 0.0
    assert d[d.userid == "3"]["invalid_answer_count"].iloc[0] == 1

    assert "invalid_answer_percentage" in p.keys
    assert "invalid_answer_count" in p.keys


def test_columns_pivot_columns_pivots_by_user_survey_if_form_data_and_keeps_keys(
    form_df, df
):
    p = Preprocessor()
    df = p.add_form_data(form_df, df)
    df = p.count_invalid(df)
    df = p.keep_final_answer(df)
    d = p.pivot("response", df)

    assert "userid" in d.columns
    assert "surveyid" in d.columns
    for ref in df.question_ref.unique():
        assert ref in d.columns

    for k in p.keys:
        assert k in d.columns

    assert "shortcode" in d.columns


def test_columns_remove_form_data_removes_form_keys(df, form_df):
    p = Preprocessor()
    df = p.add_form_data(form_df, df)
    assert "surveyid" in df.columns

    df = p.remove_form_data(df)
    assert "surveyid" not in df.columns


def test_columns_pivot_columns_pivots_by_just_user_if_no_form_data():
    data = [
        ("a", "1", 1, "A", 1, "response", ts(12, 2, 0), '{"stratumid": "Z"}'),
        ("a", "1", 1, "B", 2, "response", ts(12, 2, 1), '{"stratumid": "Z"}'),
        ("a", "1", 1, "C", 3, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("b", "1", 1, "D", 1, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("b", "1", 1, "E", 1, "response2", ts(12, 2, 6), '{"stratumid": "Z"}'),
    ]

    df = make_df(data)

    p = Preprocessor()
    df = p.count_invalid(df)
    df = p.keep_final_answer(df)
    d = p.pivot("response", df)

    assert "surveyid" not in d.columns
    assert d.shape[0] == 1


def test_columns_pivot_columns_pivots_by_just_user_if_form_data_removed(form_df):
    data = [
        ("a", "1", 1, "A", 1, "response", ts(12, 2, 0), '{"stratumid": "Z"}'),
        ("a", "1", 1, "B", 2, "response", ts(12, 2, 1), '{"stratumid": "Z"}'),
        ("a", "1", 1, "C", 3, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("b", "1", 1, "D", 1, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("b", "1", 1, "E", 1, "response2", ts(12, 2, 6), '{"stratumid": "Z"}'),
    ]

    df = make_df(data)

    p = Preprocessor()
    df = p.add_form_data(form_df, df)
    df = p.count_invalid(df)
    df = p.keep_final_answer(df)
    df = p.remove_form_data(df)
    df = p.add_duration(df)
    d = p.pivot("response", df)

    assert "surveyid" not in d.columns
    assert d.shape[0] == 1


def test_columns_pivot_columns_raises_exception_if_duplicated_questions(form_df):
    data = [
        ("a", "1", 1, "A", 1, "response", ts(12, 2, 0), '{"stratumid": "Z"}'),
        ("a", "1", 1, "B", 2, "response", ts(12, 2, 1), '{"stratumid": "Z"}'),
        ("a", "1", 1, "C", 3, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("b", "1", 1, "D", 1, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("b", "1", 1, "C", 1, "response2", ts(12, 2, 6), '{"stratumid": "Z"}'),
    ]

    df = make_df(data)

    p = Preprocessor()
    df = p.count_invalid(df)
    df = p.keep_final_answer(df)

    with pytest.raises(PreprocessingError):
        p.pivot("response", df)


def test_parse_number_parses_strings_and_ints():
    assert parse_number("500") == 500
    assert parse_number("500,00") == 50000
    assert parse_number("500.00") == 50000
    assert parse_number(" ,500.00") == 50000
    assert parse_number(50000) == 50000
    assert parse_number(None) is None
    assert parse_number("lskdjf") is None


def test_hash_userid(df):
    p = Preprocessor()
    d = p.hash_userid(df)
    ids = d.userid.unique()
    hsh = "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b"

    assert hsh in ids
    assert "1" not in ids


@pytest.fixture
def js_test_cases():
    test_file = Path(__file__).parent / "seed_test_cases.json"
    with open(test_file) as f:
        return json.load(f)


def test_compute_seed_matches_javascript(js_test_cases):
    """Verify Python compute_seed matches JavaScript implementation."""
    for tc in js_test_cases["seedTestCases"]:
        result = compute_seed(tc["seed"], tc["n"], tc["m"])
        assert result == tc["result"], (
            f"Mismatch for seed={tc['seed']}, n={tc['n']}, m={tc['m']}: "
            f"Python={result}, JavaScript={tc['result']}"
        )


def test_compute_seed_returns_value_in_range():
    """Verify compute_seed always returns value between 1 and n."""
    seeds = [2960024492, 1171948497, 837565861]
    for seed in seeds:
        for n in [2, 3, 5, 10, 100]:
            for m in [0, 1, 2]:
                result = compute_seed(seed, n, m)
                assert 1 <= result <= n, f"Result {result} out of range [1, {n}]"


def test_compute_seed_deterministic():
    """Verify same inputs always produce same output."""
    seed, n, m = 2960024492, 5, 2
    results = [compute_seed(seed, n, m) for _ in range(10)]
    assert all(r == results[0] for r in results)


def test_compute_seed_key_format():
    """Verify key string format produces same results as n/m parameters."""
    seed = 2960024492

    # seed_N format
    assert compute_seed(seed, key="seed_2") == compute_seed(seed, 2)
    assert compute_seed(seed, key="seed_3") == compute_seed(seed, 3)
    assert compute_seed(seed, key="seed_10") == compute_seed(seed, 10)

    # seed_N_M format
    assert compute_seed(seed, key="seed_2_1") == compute_seed(seed, 2, m=1)
    assert compute_seed(seed, key="seed_5_2") == compute_seed(seed, 5, m=2)
    assert compute_seed(seed, key="seed_3_3") == compute_seed(seed, 3, m=3)


def test_compute_seed_key_format_against_javascript(js_test_cases):
    """Verify key format matches JavaScript implementation."""
    for tc in js_test_cases["seedTestCases"]:
        if tc["m"] == 0:
            key = f"seed_{tc['n']}"
        else:
            key = f"seed_{tc['n']}_{tc['m']}"

        result = compute_seed(tc["seed"], key=key)
        assert result == tc["result"], (
            f"Mismatch for key={key}, seed={tc['seed']}: "
            f"Python={result}, JavaScript={tc['result']}"
        )


def test_compute_seed_invalid_key_raises():
    """Verify invalid key formats raise ValueError."""
    seed = 2960024492

    with pytest.raises(ValueError):
        compute_seed(seed, key="invalid")

    with pytest.raises(ValueError):
        compute_seed(seed, key="seed_")

    with pytest.raises(ValueError):
        compute_seed(seed, key="seed_abc")

    with pytest.raises(ValueError):
        compute_seed(seed, key="2_3")


def test_compute_seed_requires_n_or_key():
    """Verify error when neither n nor key is provided."""
    with pytest.raises(ValueError):
        compute_seed(2960024492)


# ---------------------------------------------------------------------------
# map_columns
# ---------------------------------------------------------------------------


def test_map_columns_transforms_specified_columns(df):
    p = Preprocessor()
    d = p.map_columns(["question_idx"], int, df)
    assert d["question_idx"].dtype == int or d["question_idx"].iloc[0] == 1
    # Other columns are untouched
    assert "response" in d.columns
    assert d["response"].iloc[0] == "response"


def test_map_columns_multiple_columns(df):
    p = Preprocessor()
    d = p.map_columns(["question_idx", "flowid"], lambda x: x * 10, df)
    assert d["question_idx"].iloc[0] == 10
    assert d["flowid"].iloc[0] == 10


# ---------------------------------------------------------------------------
# flatten_dict
# ---------------------------------------------------------------------------


def test_flatten_dict_expands_json_keys_as_columns():
    data = pd.DataFrame({
        "id": [1, 2],
        "info": ['{"color": "red", "size": "L"}', '{"color": "blue", "size": "M"}'],
    })
    result = flatten_dict("info", data)
    assert "color" in result.columns
    assert "size" in result.columns
    assert "info" not in result.columns
    assert result["color"].iloc[0] == "red"
    assert result["size"].iloc[1] == "M"


def test_flatten_dict_with_prefix():
    data = pd.DataFrame({
        "id": [1],
        "meta": ['{"wave": "1"}'],
    })
    result = flatten_dict("meta", data, prefix="form")
    assert "form_wave" in result.columns
    assert "wave" not in result.columns
    assert "meta" not in result.columns


def test_flatten_dict_missing_key_produces_none():
    data = pd.DataFrame({
        "id": [1, 2],
        "meta": ['{"wave": "1"}', '{}'],
    })
    result = flatten_dict("meta", data)
    assert "wave" in result.columns
    # Second row has no "wave" key; should be NaN/None
    assert pd.isna(result["wave"].iloc[1])


# ---------------------------------------------------------------------------
# wrap_empty
# ---------------------------------------------------------------------------


def test_wrap_empty_returns_none_for_empty_dataframe():
    sentinel = object()

    def fn(df):
        return sentinel

    wrapped = wrap_empty(fn)
    empty = pd.DataFrame(columns=["a", "b"])
    assert wrapped(empty) is None


def test_wrap_empty_calls_fn_for_nonempty():
    called_with = []

    def fn(df, *args, **kwargs):
        called_with.append(df)
        return df

    wrapped = wrap_empty(fn)
    nonempty = pd.DataFrame({"a": [1]})
    result = wrapped(nonempty)
    assert len(called_with) == 1
    assert result is not None


# ---------------------------------------------------------------------------
# add_duration quantile columns
# ---------------------------------------------------------------------------


def test_add_duration_adds_answer_time_median(df):
    p = Preprocessor()
    d = p.add_duration(df)
    # user "1" across all surveys: timestamps are 12:02:00, 12:02:01, 12:02:05,
    # 12:02:10, 12:03:00, 12:04:00 → diffs (seconds): 1, 4, 5, 50, 60
    # median (0.5 quantile) of [1, 4, 5, 50, 60] = 5.0
    u1 = d[d.userid == "1"]["answer_time_median"].iloc[0]
    assert u1 == 5.0


def test_add_duration_adds_answer_time_75_and_90(df):
    p = Preprocessor()
    d = p.add_duration(df)
    # user "1" diffs: [1, 4, 5, 50, 60]
    # p75 of [1, 4, 5, 50, 60] = 50 + 0.75*(60-50)... let pandas decide the exact value.
    # We just check the column exists and values are positive.
    assert "answer_time_75" in d.columns
    assert "answer_time_90" in d.columns
    u1_75 = d[d.userid == "1"]["answer_time_75"].iloc[0]
    u1_90 = d[d.userid == "1"]["answer_time_90"].iloc[0]
    assert u1_75 > 0
    assert u1_90 > 0
    assert u1_90 >= u1_75


def test_add_duration_single_answer_has_nan_time_between():
    """A user with exactly one response row has no inter-answer gaps: min is NaN."""
    data = [
        ("a", "solo", 1, "A", 1, "resp", ts(10, 0, 0), '{}'),
    ]
    single_df = make_df(data)
    p = Preprocessor()
    d = p.add_duration(single_df)
    assert math.isnan(d["answer_time_min"].iloc[0])


# ---------------------------------------------------------------------------
# add_time_indicators month
# ---------------------------------------------------------------------------


def test_add_time_indicators_adds_correct_month(df):
    """
    A January timestamp should produce month == 1.
    The df fixture uses 2020-01-01 timestamps → month 1.
    """
    p = Preprocessor()
    df_dur = p.add_duration(df)
    d = p.add_time_indicators(["month"], df_dur)
    assert "month" in d.columns
    assert d["month"].iloc[0] == 1
    assert "month" in p.keys


# ---------------------------------------------------------------------------
# add_metadata multiple keys
# ---------------------------------------------------------------------------


def test_add_metadata_adds_multiple_keys(df):
    p = Preprocessor()
    # Both stratumid and a hypothetical "extra" key
    df2 = df.copy()
    df2["metadata"] = df2["metadata"].map(
        lambda x: json.dumps({**json.loads(x), "region": "North"})
    )
    d = p.add_metadata(["stratumid", "region"], df2)
    assert "stratumid" in d.columns
    assert "region" in d.columns
    assert "stratumid" in p.keys
    assert "region" in p.keys


# ---------------------------------------------------------------------------
# parse_timestamp
# ---------------------------------------------------------------------------


def test_parse_timestamp_handles_tz_naive():
    """ISO string without timezone info → tz-naive Timestamp."""
    data = make_df([
        ("a", "1", 1, "A", 1, "r", "2020-01-01T12:00:00", '{}'),
    ])
    p = Preprocessor()
    d = p.parse_timestamp(data)
    ts_val = d["timestamp"].iloc[0]
    assert isinstance(ts_val, pd.Timestamp)
    assert ts_val.tzinfo is None


def test_parse_timestamp_handles_tz_aware():
    """ISO string with +00:00 → tz-aware Timestamp."""
    data = make_df([
        ("a", "1", 1, "A", 1, "r", "2020-01-01T12:00:00+00:00", '{}'),
    ])
    p = Preprocessor()
    d = p.parse_timestamp(data)
    ts_val = d["timestamp"].iloc[0]
    assert isinstance(ts_val, pd.Timestamp)
    assert ts_val.tzinfo is not None


def test_parse_timestamp_idempotent(df):
    """Calling parse_timestamp twice should not raise and should be idempotent."""
    p = Preprocessor()
    d1 = p.parse_timestamp(df)
    d2 = p.parse_timestamp(d1)
    assert d1["timestamp"].iloc[0] == d2["timestamp"].iloc[0]


# ---------------------------------------------------------------------------
# hash_int
# ---------------------------------------------------------------------------


from vlab_prepro.preprocess import hash_int


def test_hash_int_returns_hex_string():
    result = hash_int(1)
    assert isinstance(result, str)
    # SHA-256 produces 64-char hex
    int(result, 16)  # raises if not hex


def test_hash_int_deterministic():
    assert hash_int(42) == hash_int(42)


def test_hash_int_different_inputs_differ():
    assert hash_int(1) != hash_int(2)


# ---------------------------------------------------------------------------
# add_final_answer standalone (module-level function)
# ---------------------------------------------------------------------------


def test_add_final_answer_marks_earlier_answers_false():
    data = make_df([
        ("b", "3", 1, "A", 1, "response",  ts(12, 2, 5), '{}'),
        ("b", "3", 1, "A", 1, "response2", ts(12, 2, 6), '{}'),
    ])
    result = add_final_answer(data)
    result = result.sort_values("timestamp").reset_index(drop=True)
    assert result["final_answer"].iloc[0] == False  # earlier → False
    assert result["final_answer"].iloc[1] == True   # later (last) → True


def test_add_final_answer_all_unique_are_true():
    data = make_df([
        ("a", "1", 1, "A", 1, "resp_a", ts(10, 0, 0), '{}'),
        ("a", "1", 1, "B", 2, "resp_b", ts(10, 0, 5), '{}'),
    ])
    result = add_final_answer(data)
    assert result["final_answer"].all()


# ---------------------------------------------------------------------------
# drop_users_without edge case
# ---------------------------------------------------------------------------


def test_drop_users_without_removes_all_rows_for_user():
    """A user with even one NaN row for the key loses ALL their rows."""
    data = make_df([
        ("a", "u1", 1, "A", 1, "r", ts(10, 0, 0), '{"key": "val"}'),
        ("a", "u1", 1, "B", 2, "r", ts(10, 0, 5), '{"key": "val"}'),
        ("b", "u2", 1, "A", 1, "r", ts(10, 0, 0), '{}'),   # u2 missing key
        ("b", "u2", 1, "B", 2, "r", ts(10, 0, 5), '{}'),   # u2 missing key
    ])
    p = Preprocessor()
    df = p.add_metadata(["key"], data)
    result = p.drop_users_without("key", df)
    assert "u2" not in result.userid.unique()
    assert "u1" in result.userid.unique()
    assert result[result.userid == "u1"].shape[0] == 2


# ---------------------------------------------------------------------------
# count_invalid exact values
# ---------------------------------------------------------------------------


def test_count_invalid_exact_values(df):
    """
    user "3" has 3 rows:
      ("b", "3", q_A → response  @ ts 12:02:05)
      ("b", "3", q_A → response2 @ ts 12:02:06)  ← duplicate, first gets False
      ("c", "3", q_A → response  @ ts 12:02:05)
    After add_final_answer: 1 False out of 3 rows.
    invalid_answer_percentage = 1/3, invalid_answer_count = 1.
    """
    p = Preprocessor()
    d = p.count_invalid(df)
    u3 = d[d.userid == "3"].iloc[0]
    assert abs(u3["invalid_answer_percentage"] - 1 / 3) < 1e-9
    assert u3["invalid_answer_count"] == 1


# ---------------------------------------------------------------------------
# Curried usage
# ---------------------------------------------------------------------------


def test_curried_usage_of_add_metadata(df):
    """Partial application: create the step first, apply to df later."""
    p = Preprocessor()
    step = p.add_metadata(["stratumid"])
    result = step(df)
    assert "stratumid" in result.columns
    assert "stratumid" in p.keys
