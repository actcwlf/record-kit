import pytest

import tempfile
from record_kit import Recorder, Record
import argparse


@pytest.fixture
def records_dir():
    r_d = tempfile.TemporaryDirectory()
    yield r_d.name
    r_d.cleanup()


def test_create_record(records_dir):
    print(records_dir)

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
    print(records_dir)

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
