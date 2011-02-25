from __future__ import print_function

def find_factors(num):

  # we take advantage of the fact that (i +1)**2 = i**2 + 2*i +1
  i, sqi = 1, 1
  while sqi <= num+1:
      sqi += 2*i + 1
      i += 1  

      k = 0
      while not num % i:
          num /= i
          k += 1

      yield i,k

def print_factors(num_fac):
    if num_fac[1] > 1:
        print(str(num_fac[0]) + "**" + str(num_fac[1]),end = " ")
    else:
        print(num_fac[0],end=" ")

def factorise(value):
    try:                             #check if num is an integer
        num = int(value)             #with exceptions (see later)
        if num != float(value):     
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("Can only factorise an integer")

    factor_list = list(find_factors(num))
    def get_power(pair): return pair[1]
    factor_list = filter(get_power, factor_list)
    if factor_list:
       print(num, end=" ")
       map(print_factors, factor_list)
       print("")
    else:
        print("PRIME")
    
    
    
