.. highlight:: python

About Python
==============================

What is Python
-----------------------------
Python is a high level interpreted object oriented (OO) language.
It's main field of application is in web design and scripting. 

It was invented by Guido VanRossum in the end of the 80's and the begin of
the 90's [#]_. The name was derived of the *Monty Python's Flying Curcus*
show.

In the last five years there was a huge development of mathematical tools
and libraries for Python. Actually it seems that there is no particular reason for 
this one could see it as a phenomen, or as a trend. But in the meantime the currently
available Python projects reached now dimension that make them vaiable alternatives 
for the "classical" mathematical languages like Matlab or Mathematica.

Also there are now some very useful tools for code optimisation available like *Cython*,
that makes it possible to compile your Python Code to *C* and make it up to 1000x faster,
than normal Python code. 

There are several versions of Python interpreters. The interpreter I
refer here as Python is *CPython*, the first interpreter. The CPython interpreter is written,
as the name says, in C. The reason for this choice is, that many
numerical tools are using C bindings, and Cython also works currently
only on CPython. There are also several other 
Python Interpreters like Jython (written in Java), PyPy (written in
Python), or IronPython (written in C#) available. 

Why Python?
-----------------------------

* Intuitive Syntax
* Simple
* An easy to learn language.
* Object oriented.
* Multi paradigm (OO, imperative functional)
* Fast (if used with brains).
* Rapid development.
* A common language, so you will find answers to your problem.
* Many nice tools which makes your life easier (like Sphinx, which I use to write this report)


Get Python
------------------------------
The programs and packages used here are all open source, so they can be obtained freely.
Most Linux distributions already ship Python, because many scripts are written in Python.
See also the *Python* project page for further information [#]_ . 

For using Python I personally recommend Linux or a virtual machine
with Linux, because it's much easier to install and handle (for my taste).
But there is currently a .Net Python under development named
IronPython [#]_.
Not all packages from classical CPython are currently working on IronPython
(including NumPy), but there exists IronClad [#]_ which should make it 
possible to use these CPython modules in IronPython.

An easy way to obtain Python is to install *Sagemath* [#]_, which contains many useful packages extensions
and packages for mathematics.

Another possibility would be *FEMhub* which is a fork of *Sage* [#]_ . FEMhub is smaller, but more experimental than
Sage, and is aimed only for numerics. 
Some of the packages I introduce here are are currently outdated in Sage/FEMhub or not available yet. Current Versions are
available on my Google code project page [#]_.

The drawback of these distributions is that they are not available as .deb or .rpm packages. They have to be build
from source, and currently only work on Linux and other Unix type systems.
But there are precompiled binaries available. (I personally recommand to
build it from source because then many optimisation options are applied) 



.. rubric:: Links

.. [#] http://python-history.blogspot.com/2009/01/brief-timeline-of-python.html
.. [#] http://www.python.org/
.. [#] http://ironpython.net/
.. [#] http://code.google.com/p/ironclad/
.. [#] http://www.sagemath.org/
.. [#] http://www.femhub.org
.. [#] http://code.google.com/p/computational-sage/
