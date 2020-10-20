from record_kit import Recorder
import argparse

recorder = Recorder()
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
args = parser.parse_args()

recorder.write_meta(args)
recorder.write_header('epoch', 'loss', 'acc')
recorder.write_data_line(0, 0.1, 0.5)
recorder.write_data_line(1, 0.08, 0.9)