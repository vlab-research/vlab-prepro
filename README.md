# vlab_prepro

Preprocessing library for data from the Virtual Lab survey platform.

## Installation

``` shell
pip install vlab-prepro
```

## Usage

``` python
from vlab_prepro import Preprocessor
from toolz import pipe
import pandas as pd

responses = pd.read_csv('path-to-responses.csv')
forms = pd.read_csv('path-to-form-metadata.csv')

p = Preprocessor()

pipe(responses,

     # adds form-level information (shortcode) and form metadata
     p.add_form_data(forms),

     # adds the user metadat to the df, only the keys provided in the list
     p.add_metadata(['clusterid']),

     # adds information on question-answering duration:
     # the start time, the end time, total survey time,
     # quantiles of answering speed, etc. (note: this is slow!)
     p.add_duration,

     # adds week/month indicators based on when user started the form
     p.add_time_indicators(['week', 'month']),

     # adds an indicator for whether or not the answer is the final answer
     # (i.e. in the case the user answered multiple times because of
     # invalid answers)
     p.add_final_answer

     # adds the count/percentage of non-final answers, assumed to be non-valid,
     # at the user level (for each user it computes percentage based on their
     # entire history in your survey, across forms)
     p.count_invalid,

     # drops all answers except the final answer to each question
     p.keep_final_answer,

     # drops all users who every took a survey without the "clusterid" metadata value,
     # useful for dropping test users.
     p.drop_users_without('clusterid'),

     # drops all users who answered multiple surveys with the same "wave" value,
     # useful for dropping users who took the same survey twice somehow.
     p.drop_duplicated_users(['wave']),

     # pivots df, keeps all user/survey columns created earlier.
     p.pivot('translated_response')
)
```

## Computing Seed Values

The survey platform uses randomization seeds to assign respondents to treatment arms or show randomized content. Each respondent's seed is deterministically generated from their user ID and form ID, and is included in the data export.

To calculate the actual randomization value (e.g., which treatment arm a respondent was assigned to), use `compute_seed`:

```python
from vlab_prepro import compute_seed

# If your survey used seed_3 for a 3-arm trial:
df['treatment_arm'] = df['seed'].apply(lambda s: compute_seed(s, key="seed_3"))

# Or equivalently, using n parameter:
df['treatment_arm'] = df['seed'].apply(lambda s: compute_seed(s, n=3))
```

### Multiple randomizations

If your survey used multiple independent randomizations, they would have used different `seed_N_M` values where M creates distinct random sequences:

```python
# First randomization: seed_2 (coin flip for treatment/control)
df['treatment'] = df['seed'].apply(lambda s: compute_seed(s, key="seed_2"))

# Second randomization: seed_3_1 (3-way split, independent of first)
df['message_variant'] = df['seed'].apply(lambda s: compute_seed(s, key="seed_3_1"))

# Third randomization: seed_2_2 (another coin flip, independent of both above)
df['image_shown'] = df['seed'].apply(lambda s: compute_seed(s, key="seed_2_2"))
```

### Function signature

```python
compute_seed(seed: int, n: int = None, m: int = 0, *, key: str = None) -> int
```

- `seed`: The base seed from your data export (32-bit integer)
- `n`: Range for result (returns 1 to n inclusive)
- `m`: Rehash count for independent random values (default 0)
- `key`: Alternative format string like `"seed_3"` or `"seed_3_1"`

Returns an integer from 1 to n (inclusive).
