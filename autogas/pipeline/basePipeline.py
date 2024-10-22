class BasePipeLine:
    """ Base class for all pipelines. """

    def __init__(self, *args, **kwargs):
        self._modules_operations = {}
        self._modules_probability = []

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        pass

    def __len__(self):
        return len(self._modules_probability)

    def __getitem__(self, idx):
        return self._modules_operations[idx]

    @property
    def identity(self):
        pass

    @identity.setter
    def identity(self, value):
        pass
