from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from vlab_prepro import PreprocessingError, Preprocessor, parse_number


def ts(h, m, s):
    return datetime(2020, 1, 1, h, m, s).replace(tzinfo=timezone.utc).isoformat()


def dt(h, m, s):
    return pd.Timestamp(datetime(2020, 1, 1, h, m, s), tz="UTC")


@pytest.fixture
def df():
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

    data = [
        ("a", "1", 1, "A", 1, "response", ts(12, 2, 0), '{"stratumid": "Z"}'),
        ("a", "1", 1, "B", 2, "response", ts(12, 2, 1), '{"stratumid": "Z"}'),
        ("a", "1", 1, "C", 3, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("a", "1", 1, "D", 4, "response", ts(12, 2, 10), '{"stratumid": "Z"}'),
        ("b", "1", 1, "A", 1, "response", ts(12, 2, 0), '{"stratumid": "Z"}'),
        ("b", "1", 1, "B", 2, "response", ts(12, 3, 0), '{"stratumid": "Z"}'),
        ("a", "2", 1, "A", 1, "response", ts(12, 2, 0), '{"stratumid": "X"}'),
        ("a", "2", 1, "B", 2, "response", ts(12, 2, 5), '{"stratumid": "X"}'),
        ("c", "2", 2, "C", 2, "response", ts(12, 2, 5), "{}"),
        ("b", "3", 1, "A", 1, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
        ("b", "3", 1, "A", 1, "response2", ts(12, 2, 6), '{"stratumid": "Z"}'),
        ("c", "3", 1, "A", 1, "response", ts(12, 2, 5), '{"stratumid": "Z"}'),
    ]

    return pd.DataFrame(data, columns=columns)


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


def test_add_duration_adds_min_answer_time(df):
    p = Preprocessor()
    d = p.add_duration(df)
    assert d["answer_time_min"].iloc[0] == 1.0
    assert np.isclose(d["answer_time_min"].iloc[5], 60.0)
    assert d["answer_time_min"].iloc[7] == 5.0


def test_add_duration_adds_survey_start_time(df):
    p = Preprocessor()
    d = p.add_duration(df)
    assert d["survey_start_time"].iloc[0] == dt(12, 2, 0)
    assert d["survey_start_time"].iloc[7] == dt(12, 2, 0)


def test_add_duration_adds_survey_duration(df):
    p = Preprocessor()
    d = p.add_duration(df)
    assert d["survey_duration"].iloc[0] == 10.0
    assert d["survey_duration"].iloc[7] == 5.0


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


def test_columns_pivot_columns_pivots_by_user_survey_and_keeps_keys(form_df, df):
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
    hsh = '6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b'

    assert hsh in ids
    assert '1' not in ids
