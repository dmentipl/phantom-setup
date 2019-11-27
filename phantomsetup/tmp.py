

    def __add__(self, other):
        """Add Particles objects together."""
        if set(self.arrays.keys()) != set(other.arrays.keys()):
            raise ValueError('Both Particles must have the same arrays')
        for (key, val1), val2 in zip(self.arrays.items(), self.arrays.values()):
            if val1.shape[1:] != val2.shape[1:]:
                raise ValueError(f'{key} has different shape')
        for (key, val1), val2 in zip(self.arrays.items(), self.arrays.values()):