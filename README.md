# record-kit
A simple record toolkit for experiment.

##  Installation
```bash
pip install record-kit
```

## Usage
`experiment.py`

```python
from record_kit import Recorder                      # use Recorder to track experiment data
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
args = parser.parse_args()

recorder = Recorder()                                # init Recorder
recorder.write_meta(args)                            # save arguments like hyperparameters
recorder.write_header('epoch', 'loss', 'acc')        # specify table header
recorder.write_data_line(0, 0.1, 0.5)                # save data
recorder.write_data_line(1, 0.08, 0.9)
```

generated record in `./records/record-timestamp.md`
