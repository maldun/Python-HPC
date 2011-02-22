First steps in Python
=========================================

The "Goodbye World" program.
-----------------------------------------
In the old tradition of the "<insert Language here> for Dummies" books, 
we start with the "Goodbye World" program.

#. Make a file goodbye_world.py (or what name you like).
#. Open your Editor.
#. Write::
    
    print("Goodbye World!")

#. Execute::

    python goodbye_world.py

   and you get the output::

    Goodbye World!

Thats all!

**Remark:** If you use Sage as your Python interpreter, simply start the program with ::
    
    sage goodbye_world.py

or ::

    sage -python goodbye_world.py
    
Alternatively you can do this directly in the interpreter.
#. Open a shell
#. Type::
  
  python
  
#. Write::
  
  >>> print("Goodbye World")

and press enter. 

Basic datatpypes 
---------------------------------------------

Numbers
"""""""""""""""""""""""""""""""""""""""""""""

You can represent numbers in many ways::
  
  1
  
is the **integer** one.

::
  
  1.0
  
is the **float**  one.

::
  
  1L
  
represents the **long int** one.

There is also a representation for floats with exponential::
  
  1e3
  
which is thousand, or complex numbers::
  
  1 + 3j
  

Strings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



Basic arithmetics
---------------------------------------------------------------

Of course you can use your Python interpreter as a calculator.
Simply call 
::
  
  python
  
and then try for example::
  
  >>> 1+1
  2                                                                                                                                                                                   
  >>> 2*3
  6                                                                                                                                                                                   
  >>> 3-2                                                                                                                                                                             
  1                                                                                                                                                                                   
  >>> 1+1
  2                                                                                                                                                                                   
  >>> 1-1                                                                                                                                                                             
  0                                                                                                                                                                                   
  >>> 2*3                                                                                                                                                                             
  6
  
Division is a little more tricky in Python::
  
  >>> 1/2
  0

What happened here: A division between two integers return an integer, and Python simply returns the floor.
So taking negative numbers it works in the other direction::
  
  >>> -5/2
  -3


Notes on the syntax
---------------------------------------------

Intendation for organising blocks of codes
"""""""""""""""""""""""""""""""""""""""""""""

Codes of blocks are, unlike other programming languages like C++
not organized with parantheses but with indentation. I.e. it looks
like the following::

    Code outside Block

    <statement> <identifier(s)> :
        Code in block 1
        Code in block 1
        ...
        <statement2> <id> :
            Code in block 2 
            Code in block 2
            ...
        
        Code in block 1
        Code in block 1

        <statement3 <id3> :
            Code in block 3

        Code in block 1

    Code outside Block
        
This sounds for many confusing at the beginning (including myself),
but actually it is not. 
After writing some code (with a good editor!) one get's
used to this very quickly.
Try it yourself: After a week or even a month 
writing code in Python, go back to Matlab or C.

The benefit of this is, that the code is much more readible,
and a good programmer makes indentation nevertheless.
It's also helpful for debugging: If you make an indentation error
the interpreter knows where it happend, if you forget an **end** or
an **}** the compiler often points you to a line number anywere in the code.

**Important note:** You can choose the type of indentation as you wish.
One, two, three, four,... 2011 whitespaces, or tabulators. **But** you should
never mix whitespaces with tabulators! This will result in an error.
     
The semicolon
""""""""""""""""""""""""""""""""""""""""""""""""

Generally you don't need a semicolon, and often you don't use it.
It's usage is for putting more than one statment in a line.For example::
  
  1+1; 2+2
  


The print statement
---------------------------------------------

We start here with some explainations of the print statement.

We can print 
 