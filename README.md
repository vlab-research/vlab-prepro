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

     # adds the percentage of non-final answers, assumed to be non-valid,
     # at the user level (for each user it computes percentage based on their
     # entire history in your survey, across forms)
     p.add_percentage_valid,

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
