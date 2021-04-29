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
     p.add_form_data(forms),
     p.add_metadata(['clusterid']),
     p.add_duration,
     p.add_time_indicators(['week', 'month']),
     p.add_percentage_valid,
     p.keep_final_answer,
     p.drop_users_without('clusterid'),
     p.drop_duplicated_users(['wave']),
     p.pivot('transalted_response')
)
```
