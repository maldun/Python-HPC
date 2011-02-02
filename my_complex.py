class my_complex:

    nr_instances = 0

    def __init__(self,re,im):
        """The init method serves as constructor"""
        my_complex.nr_instances += 1
        self.re = re
        self.im = im

    def abs(self):
        """Calculates the absolute value"""
        return self.re**2 + self.im**2
