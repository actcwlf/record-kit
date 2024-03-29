# record-kit
A simple record toolkit generating human-readable data record in experiments.

##  Installation
```bash
pip install record-kit
```

## Usage
Suppose you have an experiment script like this:
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
args = parser.parse_args()

def train_ml_model():
    for i in range(2):
        epoch, loss, acc = i, 0.1, 0.5
```

Using record-kit to track these generated data is quite easy:
```python
from record_kit import Recorder, Record             # use Recorder to track experiment data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
args = parser.parse_args()


def train_ml_model():
    for i in range(2):
        epoch, loss, acc = i, 0.1, 0.5
        recorder.add_data(epoch, loss, acc)   # save data


# init Recorder
recorder = Recorder()                              # init Recorder
recorder.set_meta(args)                            # save arguments like hyperparameters
recorder.set_header('epoch', 'loss', 'acc')        # specify table header

# parse record
record = Record(recorder) # or record = Record('path/to/record')
print(record.meta)
print(record.data)
```
output
```bash
{'lr': 0.1}
   epoch  loss  acc
0    0.0  0.10  0.5
1    1.0  0.08  0.9
```

generated record in `./records/record-timestamp.md`

![record_example](https://github.com/actcwlf/record-kit/blob/main/docs/record.png)
## Record Format
A typical record includes two sections `Meta` and `Data`.
```markdown
# record-20201020-154912
## Meta
| key | value | type |
| :---: | :---: | :---: |
| lr | 0.1 | float |
## Data
| epoch | loss | acc |
| :---: | :---: | :---: |
| 0 | 0.1 | 0.5 |
| 1 | 0.08 | 0.9 |
```
## API
### `Recorder`
#### `Recorder(file_name='record', records_dir='./records', with_timestamp=True)`
Initialize a Recorder instance

#### `Recorder.set_meta(dict_like_object)`
Add meta info to the record. Note meta info can only be added once. 
You may use `dict` or `argparse.Namespace`(return type of `parser.parse_args()`) as its parameter.

#### `Recorder.set_header(*args)`
Specify header of data table in Data section. The header can only be added once. 

#### `Recorder.add(*args)`
Add a data line to the table in `Data` section.
The number of parameters in should be consistent with `Recorder.set_header(*args)`

### `Record`
#### `Record(Recorder_or_path_str)`
Load & parse record

#### `Record.meta -> dict`
Get meta info of record.

#### `Record.data -> pandas.DataFrame`
Get data table of record.

