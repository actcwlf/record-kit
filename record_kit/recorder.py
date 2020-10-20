import pathlib
import time
import os


def check_suffix(file_name):
    suffixes = file_name.suffixes
    if len(suffixes) < 1 or suffixes[-1] != '.md':
        suffixes.append('.md')
        suffix = ''.join(suffixes)
        file_name.with_suffix(suffix)
    return str(file_name.absolute())


class LogBase:
    def __init__(self, file_path):
        self.file_path = file_path

    def append(self, s):
        with self.file_path.open("a") as f:
            f.write(s)

    def append_line(self, s):
        with self.file_path.open("a") as f:
            f.write(s + "\n")


class Recorder(LogBase):
    def __init__(self, file_name='record', records_dir='./records', with_timestamp=True):
        self.records_dir = pathlib.Path(records_dir)
        os.mkdir(self.records_dir)
        if with_timestamp:
            timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
            file_name += '-' + timestamp
        self.record_name = self.records_dir.joinpath(file_name + '.md')
        super().__init__(self.record_name)
        self.append_line(f"# {file_name}")

    def begin_meta(self):
        s = "## Meta\n" + \
            "| key | value |\n" + \
            "| :-: |  :-:  |\n"
        self.append(s)

    def begin_data(self):
        s = "## Data"
        self.append_line(s)

    def write_meta(self, key, value):
        s = f"| {key} | {str(value)} |"
        self.append_line(s)

    def write_header(self, *args):
        length = len(args)
        args = map(str, args)
        s = " | ".join(args)
        s = "| " + s + " |"
        s2 = "| :-: " * length + "|"
        self.append_line(s)
        self.append_line(s2)

    def write_data(self, *args):
        args = map(str, args)
        s = " | ".join(args)
        s = "| " + s + " |"
        self.append_line(s)