import os, sys

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None, mode="w"):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            try:
                os.makedirs(os.path.dirname(fpath))
            except FileExistsError:
                pass
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)


    def close_open(self):
      if self.file is not None:
        self.file.close()
        self.file = open(self.fpath, 'a')


    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()