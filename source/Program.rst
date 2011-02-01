Programming in Python
===========================================

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
    print("Start")

**Remark:** To get out more performance of your Python code use
``xrange`` instead of range, because ``xrange`` doesn't need allocate
memory for a list. In Python 3, however, ``range`` returns an iterator
and not a list, so this is obsolete there.

See also the Python wiki [#]_ on this topic.

.. rubric:: Links

.. [#] http://wiki.python.org/moin/PythonSpeed/PerformanceTips
