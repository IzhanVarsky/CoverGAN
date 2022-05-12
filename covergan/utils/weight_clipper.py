class WeightClipper(object):

    def __init__(self, c):
        self.c = c

    def __call__(self, module):
        if self.c is not None and hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clip(-self.c, self.c)
            module.weight.data = w
