.. highlight:: rst


Introduction and personal thoughts 
=======================================

Foreword
-----------------------------------------------------------------------------------------

This short documentation was written by me for the High Performance Computing Seminar
in the winter semester 2010/11 of Prof. G. Haase at the University Graz Austria.

Actually to learn Sphinx and to make it possible for other to get a quick and dirty reference
for working with Python in mathematics and scientific computing I started to write this tech report.

I started with Python last summer, after a short introduction to the *Sage* mathematics software. One could
say it was love at first sight. I was implementing some test code with krylov methods in Matlab and Octave that
time, and was annoyed by the lack of object oriented features like abstracting and capsuling. I had the problem, that
every time I implement a new numerical scheme I have to rewrite the code of my optimisations algorithms, or at least have 
to alter it, so that every time I need to retest the new implemtation, which costs time and nerves. And since I'm a
lazy person I didn't want to do that. 

I'm now implementing my thesis code in Python, and I also work as a Sage developer in my freetime, and try to help
improving the numerics, optimisations and symbolics parts of Sage which are my personal research interests.

The Python version used here is Python 2.6. since most of the current packages are not ported to Python 3.
But I try to make them compatible with the new version so that this document don't get outdated soon.

This document is aimed towards mathematicians who want learn Python in a quick way to use it.
I have to admit that I'm a beginner too, and would be happy about
feedback on this document. I plan to extend it frequently with tricks
I learned and collected on various newsgroups. I don't know If this is
a good guide, but I hope potential readers have the same fun reading,
I had writing.


I distribute this under an open license so people can share it
freely. (See Section :ref:`license_ref` )

Stefan Reiterer, 
Graz Austria 
2011


Experiences I want to share with programming beginners
-------------------------------------------------------------------------------------

My professors in the basic programming and informatics lectures were all software devolopers, and often cursed programmers from scientific areas,
because of their complicated and often weird codes. I didn't understand their words that time, but since I'm working with libraries like
BLAS, LAPACK, ATLAS etc. I started to understand...  

It's true that the processes of software engeneering for "normal" applications and scientific computation are two different areas, but I realised in the recent
years that many people from the latter area seem to simply ignore **nearly all** basic concepts of software design and coding, and I don't know why.
Maybe it's ignorance, because many think they don't need that much programming again, or because they are simply lazy. Another reason could be 
that they are too deep into it, and think everyone else think the same way. Perhaps it has historical reasons, like in the case of BLAS,
or it's me because of my friends and education I have a different viewpoint on  that things.

Neverteless I want to use this section to give some important lectures to people, who aren't deep into programming, 
I learnt during the last 13 years since I'm started "programming" Visual Basic with 13.

The Zen of Python don't apply only to Python
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

If you type into your Python Interpretor the line
::

    import this

You will get this:

**The Zen of Python, by Tim Peters**

#. *Beautiful is better than ugly.*
#. *Explicit is better than implicit.*
#. *Simple is better than complex.*
#. *Complex is better than complicated.*
#. *Flat is better than nested.*
#. *Sparse is better than dense.*
#. *Readability counts.*
#. *Special cases aren't special enough to break the rules.*
#. *Although practicality beats purity.*
#. *Errors should never pass silently.*
#. *Unless explicitly silenced.*
#. *In the face of ambiguity, refuse the temptation to guess.*
#. *There should be one-- and preferably only one --obvious way to do it.*
#. *Although that way may not be obvious at first unless you're Dutch.*
#. *Now is better than never.*
#. *Although never is often better than *right* now.*
#. *If the implementation is hard to explain, it's a bad idea.*
#. *If the implementation is easy to explain, it may be a good idea.*
#. *Namespaces are one honking great idea -- let's do more of those!*

This is the philosophy of python and can argue about some the points,
e.g. point 13., but I can say without bad feeling, that **every**
programmer should especially keep in mind the points 1-7, which apply in every
language you will use!

Code is more often read than written
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
For every time code is written, it is read about 10 times, and
five times by yourself! If you write code use good and intuitive 
names of the variables you use, and make enough comments in your code.
One often writes code, and then have to look at it a month later, and if
you didn't a good work on naming and commenting, you will spend many ours 
on trying to understand what you have done that time. And remember: Its **your** time.
So don't do it unless you want to assure your employment.
And if you want to use short variables like *A* for a matrix make sure to mention 
that at the beginning of a function which uses these variables.
And rest assured: Using longer variable names don't cost performance.

Program design isn't a waste of time!
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Of course you don't need to design every snippet of code you do,
but at least take your time to think about the implementation, and
how you can eventually reuse it. Sometimes ten minutes of thinking
can save yourself ours of programming.

Object oriented programming abstracts away your problems
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
If one is not familiar with the paradigm of object oriented programming change this!
There are tons of books and websites on this topic.
OO programming is not a trend of the last decades, it's the way of abstract mathematics itself.
Mathematicians don't study special cases all the time. We try to exctract the very essence of a class 
of problems, and build a theory only using these fundamental properties. This makes it possible to
use theorems on huge classes of problem and not only on one.

Carefully done this saves yourself alot of programming time, because now you are able
to program your algorithms not only for some special input, but for a whole class of objects
in the literal sense.

This semester I gave also an excercise in the optimisation course, where
all the linesearch methods we implemented had to be integrated into one steepest descent algorithm.
While my students needed ours to implement this in Matlab. I only needed one half in Python, because
I simply subdivided the sub problems in classes, and had to write the framework algorithm only once. 

Premature optimisation is the rule of all evil!
"""""""""""""""""""""""""""""""""""""""""""""""""
This often cited quote of Donald E. Knuth [#]_ is true in it's very deep essence. In an everage program
there are about only 3% of critical code. But many programmers invest their time to optimise the
other 97% and wonder why their program isn't getting quicker. The only gain you get is a whole bunch
of unreadible code. I remember that I implemented an "optimized" for loop some time ago, and the only gain were
3 ms of more speed. And later when I looked on that function I had no Idea what I did that time... 

Choice of the right tools
"""""""""""""""""""""""""""""""
Since I descend a family of craftsmans, this was taught me very early. You don't want to use
a siedgehammer for hitting a tiny nail into a wall, and you don't want to use small axe to cut down a tree.   
(Well I know people who do...). And this applies for programming as well. You don"t want to write a parser in Fortran,
and you don't want to write a program for symbolic manipulation in Java. (I personally would never implement *anything* 
mathematical in Java, because it lacks some aspects like operator overloading and efficiency, but that's only a biased opinion.) 
The right choice of used languages
and tools, can heavily affect the time you need, and also your success of your projects. It often helps to
ask colleagues, teachers and Google to find the right tool. I list
some of the tools I use here. Keep always in mind that the choice of
your tools, depends also on your personal skills, and
preferences. Something that a colleague of yours like, could possible
a nuissance for yourself.

Don't use Notepad as your editor!
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A good editor is not expensive (often even free), and saves you a
whole lot of work! Good editors are for example Emacs [#]_, VIM [#]_.
A good List of editors can be found on Wikipedia. [#]_

Use version control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many, many people simply don't know there are very nice
tools to keep record of your changes, and make it possible
to redo the changes. Most common are Git [#]_, Mercurial [#]_ (which is written in Python),
or SVN [#]_. 

Use debugging tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Very good debugging tools are for example Valgrind [#]_, GDB [#]_,
and many, many more... [#]_ 

Use Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is of course only a personal recommondation. But Linux is in my opinion better
suited as development enviroment, because most things you need for programming are native, or
already integrated, and even the standard editors know syntax highlighting of the most programming
languages. Even C# is well integrated in Linux nowadays, and many useful programming tools are simply not
available in Windows (including many of the things we use here).
You don't even need to install a whole Linux distribution. Recently there was a huge development of free
Virtual Machines like Virtual Box [#]_, or projects like Wubi [#]_. And thanks to Distributions like
Ubuntu [#]_  and it's many derivatives (I use Kubuntu), or open SUSE [#]_ using Linux is nowadays possible for
normal humans too.

Not everything from Extreme Programming is that bad
""""""""""""""""""""""""""""""""""""""""""""""""""""
It is shown in many tests that applying the whole concept of XP [#]_, simply 
doesn't work in practice.
However, done with some moderation the basic concepts of extreme programing can make 
the life of a programmer much easier. I personally use this modified subset of rules:

* The project is divided into iterations.
* Iteration planning starts each iteration.
* Pair programming (at least sometimes).
* Simplicity.
* Create spike solutions to reduce risk.
* All code must have unit tests.
* All code must pass all unit tests before it  can be released/integrated.
* When a bug is found tests are created.

Examples say more than thousend words!
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Make heavy use of examples. They are a quick reference, and you
can use them for testing your code as well.

If your programs aren't understandable nobody will use them
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
...including yourself.


 
Use your brain! 
""""""""""""""""""""""""""""""""
Implicitely used in all points above, this is the most fundamental thing.  
Never simply apply concepts or techniques without thinking about the consequences,
or if they are suited for your problems. And yes I include my guidelines here as well.
I met many programmers and software developers, which studied software design, and 
how to use design tools, but never really think about the basics. Many bad design decisions
were decided this way! 

I also often hear about totally awesome newly discovered concepts, which I use in my daily basis,
because I simply don't want to do unessecary work.

.. rubric:: Links

.. [#] http://en.wikiquote.org/wiki/Donald_Knuth
.. [#] http://www.gnu.org/software/emacs/
.. [#] http://www.vim.org/
.. [#] http://en.wikipedia.org/wiki/List_of_text_editors
.. [#] http://git-scm.com/
.. [#] http://mercurial.selenic.com/
.. [#] http://subversion.apache.org/
.. [#] http://valgrind.org/
.. [#] http://www.gnu.org/software/gdb/
.. [#] http://en.wikipedia.org/wiki/Debugger
.. [#] http://www.virtualbox.org/
.. [#] http://www.ubuntu.com/desktop/get-ubuntu/windows-installer
.. [#] http://www.ubuntu.com/
.. [#] http://www.opensuse.org/de/
.. [#] http://www.extremeprogramming.org/

  
