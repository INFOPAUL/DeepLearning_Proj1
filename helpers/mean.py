class Mean:
    def __init__(self):
        self.avg = None
        self.counter = 0

    def add(self, val, weight):
        self.counter += weight
        if self.avg is None:
            self.avg = val
        else:
            delta = val - self.avg
            self.avg += delta * weight / self.counter

    def val(self):
        return self.avg