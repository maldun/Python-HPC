Programming in Python
===========================================

In this section I will give all tools for programming with
Python. The sections are ordered by programming paradigms.
If you are new to programming, the last paragraph of this section 
contains an overview of the paradigms. ( :ref:`paradgm_ref` )

Commenting in Python
-------------------------------------------

To comment out lines of codes use ``#``.

Examples::

  # I'm a comment
  
  x = x + 1 # do some stuff
  
  # bla
  # bla

.. _control_flow_ref:

Go with the control flow
-------------------------------------------

The ``if`` statement
"""""""""""""""""""""""""""""""""""""""""""

The ``if`` statement in Python is quite the way one would
expect from other languages.
As mentioned in the section :ref:`indention_ref` the ``if`` statement
has the following structure::

  if condition_is_true:
      do_something

Note the intendention!

There is also an ``else`` statement in Python::

  if condition_is_true:
      do_something
  else:
      do_something_else

Note that for the ``else`` statement the intendention rule applies
too!

There is also an ``elif`` clause short for else/if::

    if condition_is_true:
        do_something
    elif another_condition_is_true:
        do_something_different
    else:
        do_something_else

Here for example we determine the sign of a value::

  if x > 0:
      sign = 1
  elif x < 0:
      sign = -1
  else:
      sign = 0

``while`` loops
"""""""""""""""""""""""""""""""""""""""""""""

``while`` loops are also like expected::

  while condition_is_true:
      do_something

In Python while loops know alos an ``else`` statement.
It is executed when the condition is violated::

  while condition_is_true:
      do_something
  else:
      do_something

Here an example::

  k = 0
  while k < 10:
      print(k)
      k += 1
  else:
      # if k >= 10 we come into the else clause
      print("Start")

the output of this snippet is::

  0
  1
  2
  3
  4
  5
  6
  7
  8
  9
  Start

``for`` loops
"""""""""""""""""""""""""""""""""""""""""""""""""

For loops are a little bit different in Python,
because in contrast to other programming languages
``for`` iterates through a sequence/list, and not only to integers
or numbers, like in C.

A ``for`` loop looks like this::

  for x in list:
      do_something_with_x

We can use the ``range`` function (see the section about
:ref:`list_ref` ) to create a *norma;* ``for`` loop::

  for i in range(n):
      do_something_with_x

The ``for`` loop knows also an ``else`` statement. It is executed when
``for`` reaches the end of the list/sequence. 

Analogous to our ``while`` example::

  for k in range(10):
      print(k)
  else: 
      # When end of list is reached...
      print("Start")

**Remark:** To get out more performance of your Python code use
``xrange`` instead of range, because ``xrange`` doesn't need allocate
memory for a list. In Python 3, however, ``range`` returns an iterator
and not a list, so this is obsolete there.

See also the Python wiki [#]_ on this topic.

The ``break`` and ``continue`` statements
""""""""""""""""""""""""""""""""""""""""""""

The ``break`` and ``continue`` statements are borrowed from *C*.

* ``continue`` continues with the next iteration of the loop.
  For example::

    >>> k = 0
    >>> for i in range(10):
    ...     k += i
    ...     continue # Go on with next iteration
    ...     print(k) # The interpreter never reaches this line
    ... else:
    ...     print(k) # print result
    ... 
    45

* ``break`` breaks out of the smallest enclosing ``for`` or ``while``
  loop.
  Here a famous example from the official Python tutorial [#]_ ::

  >>> for n in range(2, 10):
  ...     for x in range(2, n):
  ...         if n % x == 0:
  ...             print n, 'equals', x, '*', n/x
  ...             break
  ...     else:
  ...         # loop fell through without finding a factor
  ...         print n, 'is a prime number'
  ...
  2 is a prime number
  3 is a prime number
  4 equals 2 * 2
  5 is a prime number
  6 equals 2 * 3
  7 is a prime number
  8 equals 2 * 4
  9 equals 3 * 3

The ``pass`` statement
"""""""""""""""""""""""""""""""""""""""""""

The ``pass`` statement, in fact, does nothing.
It can be used as a placeholder for functions,
or classe which have to be implemented yet.

For example the snippet
::

  while 1:
      pass

results in an endless loop, where nothing happens.  

Defininng functions
--------------------------------------------------

A function is declared with the ``def`` statement in normal Python
manner.
The statment has to be followed by an identifier
We simply start with a classical example, and give explaination later on.

The factorial would be implemented in Python that way::

  def my_factorial(n):
      """ Your documentation comes here"""
      
      k = 1
      for i in xrange(1,n+1):
          k *= i

      return k # Give back the result

The ``return`` statement
""""""""""""""""""""""""""""""""""""""""

The ``return`` statement terminate the function and returns the value.
To return more values simply use a comma::

  def f(x,y):
    return 2*x, 3*y

Python return them as a tuple::

  >>> a = f(2,3)
  >>> a
  (4, 9)

If you dont want to store them in a
tuple simple use more identifiers seperated by a comma::

  >>> b,c  = f(2,3)
  >>> b
  4
  >>> c
  9

``return`` without an expression returns ``None``

Variables (inside functions)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Variables within a function are all local, except they are 
defined outside of the code block::

  >>> x = 1      # declared outside of the function
  >>> def f():
  ...     a = 2    # declared inside of the function
  ...     print(x) # can be called within the function
  ... 
  >>> f()
  1
  >>> a # not defined outside of the function
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  NameError: name 'a' is not defined
   But you can't assign values
  to a global variable within a function


But you can't assign a global varaible a new value within a function::

  >>> x = 1
  >>> def f():
  ...     x = 2
  ...     print(x)
  ... 
  >>> f()
  2
  >>> x
  1

except you use the ``global`` statement::

  >>> global Bad    # Declare identifier as global
  >>> Bad = 1       
  >>> def f():
  ...     global Bad  # Tell the function Bad is global 
  ...     Bad = 2
  ...     print(Bad)
  ... 
  >>> Bad
  1
  >>> f()
  2
  >>> Bad
  2

but I would avoid this as much as possible...


Default values and keyword arguments
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Python allows to define functions with default values::

  >>> def answering(name, mission, answer="I dont know"):
  ...     print("What's your name?")
  ...     print(name)
  ...     print("What's your mission?")
  ...     print(mission)
  ...     if answer == "I dont know":
  ...         print(answer + " Ahhhhhhhhhh!")
  ...     else:
  ...         print(answer)
  ...         print("You may pass")
  ... 
  >>> answering("Gallahad", "The search for the holy grail")
  What's your name?
  Gallahad
  What's your mission?
  The search for the holy grail
  I dont know Ahhhhhhhhhh!
  >>> answering("Lancelot", "The search for the holy grail", "Blue")
  What's your name?
  Lancelot
  What's your mission?
  The search for the holy grail
  Blue
  You may pass

Docstrings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Docstrings are optional, and come right after the definition of the
function. A docstring is simply a string. Here is an example::

  >>> def doubling(x):
  ...   """I'm doubling stuff!
  ...      Yes it's true!"""
  ...   return 2*x
  ...  
  >>> print doubling.__doc__
  I'm doubling stuff!
       Yes it's true!

There are many powerful tools like Sphinx, where you can use your
docstrings for creating documentation of your code, or tools for
automatic testing, which read take the docstring as input.

.. _paradigm_ref:

Some words on programming paradigms
-------------------------------------------

There are several programming paradigms, and the
most common in modern programming languages are

* Imperative programming
* Functional programming
* Object oriented programming

Look at for exmaple at Wikipedia for an short 
overwiev on that topic [#]_, or a good programming book
of your choice, If you want to go deeper into that topic.

In short: 

* In *imperative programming* you define sequences of
  commands the computer should perform, with help of loops,
  control statements, and functions. The program has *states*
  which determine, what the program does, and which action to 
  perform. This is a quite natural approach to programming, because
  a human works also that way, for example: state "hunger" -> get
  food). Classical examples for such languages are *Fortran* (the 
  first high level language) or *C*.

* *Functional programming* is a little bit more artifical, 
   but often a more elegant
   approach for programming. In functional programming you define
   functions and let them operate on objects, lists, or call them
   recursivly. An example would be the *Lisp* family, which was 
   also the first one. (It's worthwile to look at *Lisp* not only
   to customize your Emacs. A good reading tip would be: Practical
   Common Lisp [#]_ ) One important benefit of functional programming
   is, that is easier to parallize. For example it's easier for the 
   compiler/interpreter to decide, when you operate with a function on a list,
   because all operations are independent anyway, than within a for
   loop where the compiler/interpreter doesn't know if there are operations  
   which could be possible connected.

* *Object oriented programming* is (dear computer scientists, don't
   send me hatemail) more a way to organize your data, and program
   than a real paradigm, and in fact you can program OO even in *C*
   with the help of structs. I already wrote a little about
   that (see :ref:`OO_ref` ), and at least for everyone who does 
   abstraction in a regular  basis this is a very intuitive concept.
   (And in fact every human does! ) 
   OO programming means to collect things, that share specific
   attributes in certain classes. And every Object that shares
   those features belongs to that class. A real world example
   would be wheels: There are big wheels, small wheels, wheels
   for snow etc. but they all share common properties that makes
   them wheels (For example they are all round, 
   and break in a regular basis). 
   
 
The good news are, that in Python you are able to work with
all three at least to some extend. (Python is more imperativ
than funcional). That means Python is a multi paradigm language.

Even if some say that one of the three is the true answer, I
personally think that all three have their benefits and drawbacks,
and thats the reason I prefer multiparadigm languages like Python, because
sometimes it is easier and more intuitive to program a functionality
in one certain way, while it's not so easy in the others.

For example I think it's easier and more elegant to write
:: 

  def f(x): return 2*x
  x = range(10)
  
  x = map(f,x)

than
::

  def f(x): return 2*x
  x = range(10)
  
  for i in x:
    x[i] = f(x[i])


and it's more intuitive and easier to write
::

  def f(x): return 2*x
  x = range(10)
  
  for i in range(0,10,2):
    x[i] = f(x[i])
  
than  
::

  def f(x): return 2*x
  x = range(10)
 
  map(lambda i: f(x[i]), range(0,10,2))
  
.. rubric:: Links

.. [#] http://wiki.python.org/moin/PythonSpeed/PerformanceTips
.. [#] http://docs.python.org/tutorial/controlflow.html
.. [#] http://en.wikipedia.org/wiki/Programming_paradigm
.. [#] http://www.gigamonkeys.com/book/
