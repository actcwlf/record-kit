import pytest

import tempfile
from record_kit import Recorder, Record
import pathlib


@pytest.fixture
def records_dir():
    r_d = tempfile.TemporaryDirectory()
    yield pathlib.Path(r_d.name)
    r_d.cleanup()


def test_create_record(records_dir):
    # print(records_dir)

    recorder = Recorder(records_dir=records_dir)


    recorder.set_meta({
        'learning_rate': 0.01,
        'epoch': 100,
        'model': 'mlp',
    })
    recorder.set_header('epoch', 'loss', 'acc')
    recorder.add_data(0, 0.1, 0.5)
    recorder.add_data(1, 0.08, 0.9)

    record = Record(recorder) # or record = Record('path/to/record')
    assert 'learning_rate' in record.meta.keys()
    assert record.meta['learning_rate'] == 0.01
    assert record.meta['epoch'] == 100
    assert record.meta['model'] == 'mlp'
    assert record.data['epoch'][0] == 0


def test_hybrid_data_type(records_dir):
    # print(records_dir)

    recorder = Recorder(records_dir=records_dir)


    recorder.set_meta({
        'learning_rate': 0.01,
        'epoch': 100,
        'model': 'mlp',
    })
    recorder.set_header('data', 'loss', 'acc')
    recorder.add_data('set1', 0.1, 0.5)
    recorder.add_data('set1', 0.08, 0.9)

    record = Record(recorder)
    assert 'learning_rate' in record.meta.keys()
    assert record.meta['learning_rate'] == 0.01
    assert record.meta['epoch'] == 100
    assert record.meta['model'] == 'mlp'
    assert record.data['data'][0] == 'set1'

sample_content = """
# record-20221003-182442
## Meta
| key | value | type |
| :---: |  :---: | :---: |
| lambda1 | 5 | int |
| lambda2 | 1 | int |

## Data
| data_set  | kernel_comp |
| :---: | :---: | 
| ACSF1 | None |
| ACSF1 | (113, 163, 23) |
"""

def test_full(records_dir):
    with open(records_dir.joinpath('sample.md'), 'w') as f:
        f.write(sample_content)
    record = Record(records_dir.joinpath('sample.md'))
