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

class my_new_complex(my_complex):

    def abs(self):
        """Calculates the absolute value"""
        return (self.re**2 + self.im**2)**0.5

class my_nice_complex(my_new_complex):

    def __add__(self,other):
        return my_nice_complex(self.re + other.re,self.im + other.im)

    def __mul__(self,other):
        return my_nice_complex(self.re*other.re - self.im*other.im,
                               self.re*other.im + self.im*other.re) 

    def __repr__(self):
        return "{0} + {1}i".format(self.re,self.im)
