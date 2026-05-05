class OutputRecorder:
    def __init__(self, target, index=None):
        self.target = target
        self.index = index


def capture_outputs(fn):
    return fn
