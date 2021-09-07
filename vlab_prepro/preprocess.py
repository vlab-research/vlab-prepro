import hashlib
import json
import logging
import re

import pandas as pd
from toolz import curry


class PreprocessingError(BaseException):
    pass


def wrap_empty(fn):
    def _wrapper(df):
        if df.shape[0] == 0:
            return None
        return fn(df)

    return _wrapper


def _flatten_dict(col, r):
    for k, v in json.loads(r[col]).items():
        r[k] = v
    return r


def flatten_dict(col, df):
    return df.apply(lambda r: _flatten_dict(col, r), 1).drop(columns=[col])


def _new_cols(left, right):
    left = {k for k in left.columns}
    return {k for k in right.columns if k not in left}


def _add_duration(df):
    df = df.sort_values("timestamp")
    df["survey_start_time"] = df.timestamp.iloc[0]
    df["survey_end_time"] = df.timestamp.iloc[-1]
    df["survey_duration"] = (
        df.timestamp.iloc[-1] - df.timestamp.iloc[0]
    ).total_seconds()

    time_to_answer = df.timestamp.diff().dt.total_seconds()
    df["answer_time_min"] = time_to_answer.min()
    df["answer_time_median"] = time_to_answer.quantile(0.5)
    df["answer_time_75"] = time_to_answer.quantile(0.75)
    df["answer_time_90"] = time_to_answer.quantile(0.90)

    return df


def add_final_answer(df):
    df = df.sort_values("timestamp")
    df["final_answer"] = True
    df.loc[
        df.duplicated(["userid", "surveyid", "question_idx"], keep="last"),
        "final_answer",
    ] = False
    return df


def _add_time_indicators(indicators, df):
    tindex = "survey_start_time"

    if tindex not in df.columns:
        raise KeyError(f"Could no find time index: {tindex} in dataframe columns")

    for name, frame, fn in indicators:
        df = (
            df.resample(frame, on=tindex)
            .apply(wrap_empty(lambda d: d.assign(**{name: fn(d[tindex].iloc[0])})))
            .reset_index(drop=True)
        )

    return df


def drop_duplicated_users(form_keys, df):
    # form_keys should uniquely identify your form
    # (i.e. shortcode! Or, if there are multiple shortcodes that shouldn't
    # be taken twice, some other metadata that the forms have to identify them)

    # multiple flowids means user came back and took form again
    keys = ["userid"] + list(form_keys)
    duplicated_users = (
        df.groupby(keys)
        .filter(lambda df: df.flowid.unique().shape[0] > 1)
        .userid.unique()
    )

    logging.warning(f"Removing {len(duplicated_users)} users for duplication.")

    return df[~df.userid.isin(duplicated_users)]


def parse_number(s):
    """Follows similar validation rules to the chatbot number validator

    Note: these rules are pretty loose and maybe result in silly numbers.

    """
    try:
        s = re.sub(",", "", s)
        s = re.sub(r"\.", "", s)
        s = s.strip()
        return int(s)
    except TypeError:
        return s
    except ValueError:
        return None


def hash_int(i):
    b = str(i).encode("ASCII")
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


class Preprocessor:
    def __init__(self):
        self.keys = {"userid"}
        self.form_df = None

    @curry
    def add_form_data(self, form_df, df):
        new_form_df = flatten_dict("metadata", form_df)
        self.form_df = new_form_df
        self.keys = self.keys | set(new_form_df.columns)
        return df.merge(new_form_df, on="surveyid")

    @curry
    def remove_form_data(self, df):
        if self.form_df is None:
            raise PreprocessingError("No form data to remove from the dataframe.")

        df = df.drop(self.form_df.columns, axis=1)
        self.keys = self.keys - set(self.form_df.columns)
        self.form_df = None
        return df

    @curry
    def add_metadata(self, keys, df):
        for key in keys:
            df[key] = df.metadata.map(lambda x: json.loads(x).get(key))
            self.keys.add(key)
        return df

    @curry
    def parse_timestamp(self, df):

        # NOTE: If ISO has TZ info, then it will be TZ aware, otherwise
        # it will be TZ naive.
        return df.assign(timestamp=df.timestamp.map(lambda x: pd.Timestamp(x)))

    @curry
    def add_duration(self, df):
        if not pd.api.types.is_datetime64_dtype(df.timestamp):
            df = self.parse_timestamp(df)

        df = (
            df.groupby(list(self.keys), dropna=False)
            .apply(_add_duration)
            .reset_index(drop=True)
        )

        self.keys = self.keys | {
            "survey_start_time",
            "survey_end_time",
            "survey_duration",
            "answer_time_min",
            "answer_time_median",
            "answer_time_75",
            "answer_time_90",
        }

        return df

    @curry
    def add_time_indicators(self, indicators, df):
        if not pd.api.types.is_datetime64_dtype(df.timestamp):
            df = self.parse_timestamp(df)

        lookup = {
            "week": ("week", "1w", lambda i: i.week),
            "month": ("month", "1m", lambda i: i.month),
        }

        inds = [lookup[i] for i in indicators]

        for i in indicators:
            self.keys.add(i)

        return _add_time_indicators(inds, df)

    @curry
    def add_final_answer(self, df):
        return add_final_answer(df)

    @curry
    def keep_final_answer(self, df):
        if "final_answer" not in df.columns:
            df = self.add_final_answer(df)

        return df[df.final_answer].reset_index(drop=True)

    @curry
    def count_invalid(self, df):
        if "final_answer" not in df.columns:
            df = self.add_final_answer(df)

        df = (
            df.groupby("userid")
            .apply(
                lambda df: df.assign(
                    invalid_answer_percentage=(~df.final_answer).mean(),
                    invalid_answer_count=df.final_answer.count()
                    - df.final_answer.sum(),
                )
            )
            .reset_index(drop=True)
        )

        self.keys.add("invalid_answer_percentage")
        self.keys.add("invalid_answer_count")

        return df

    @curry
    def drop_users_without(self, metadata_key, df):
        """Used to drop testers"""

        if metadata_key not in df.columns:
            raise PreprocessingError(
                f"Dataframe does not have column {metadata_key}."
                " Maybe consider running add_metadata first?"
            )

        testers = df[df[metadata_key].isna()].userid.unique()
        df = df[~df.userid.isin(testers)].reset_index(drop=True)

        logging.warning(
            f"Removing {len(testers)} users who answered a survey without"
            f" a value for {metadata_key} in the metadata"
        )

        return df

    @curry
    def drop_duplicated_users(self, form_keys, df):
        return drop_duplicated_users(form_keys, df)

    @curry
    def pivot(self, answer_column, df):
        keys = self.keys

        if "surveyid" not in keys:
            logging.warning(
                "Pivoting without survey information. "
                "Make sure question_refs are unique across surveys "
                "as all surveys will be collapsed"
            )

        try:
            return (
                df.pivot(keys, "question_ref", answer_column)
                .reset_index()
                .sort_values(["userid"])
            )
        except ValueError as e:
            raise PreprocessingError(
                "Could not pivot. Potentially you should use add_form_data "
                "to add surveyid and ensure that each user/survey is a unique line "
                "or remove duplicated questions or duplicated users"
            ) from e

    @curry
    def map_columns(self, cols, fn, df):
        return df.assign(**{col: df[col].map(fn) for col in cols})

    @curry
    def hash_userid(self, df):
        return df.assign(userid=df.userid.map(hash_int))
