maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
maldun@hexenkessel:~/tex/hpc/docu$ sage -python -build_ext --inplace
Unknown option: -l
usage: python [option] ... [-c cmd | -m mod | file | -] [arg] ...
Try `python -h' for more information.
maldun@hexenkessel:~/tex/hpc/docu$ sage -python -ext_build --inplace
Unknown option: -e
usage: python [option] ... [-c cmd | -m mod | file | -] [arg] ...
Try `python -h' for more information.
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py -build_ext --inplace
usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: setup.py --help [cmd1 cmd2 ...]
   or: setup.py --help-commands
   or: setup.py cmd --help

error: option -b not recognized
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py -build --inplace
usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: setup.py --help [cmd1 cmd2 ...]
   or: setup.py --help-commands
   or: setup.py cmd --help

error: option -b not recognized
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
gcc -pthread -shared build/temp.linux-i686-2.6/my_matvec.o -L/home/maldun/sage/sage-4.6.1/local/lib -lpython2.6 -o /home/maldun/tex/hpc/docu/my_matvec.so
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
     ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/my_matvec.pyx:1:6: Syntax error in simple statement list
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
my_matvec.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
     ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/my_matvec.pyx:1:6: Syntax error in simple statement list
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
my_matvec.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
     ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/my_matvec.pyx:1:6: Syntax error in simple statement list
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
my_matvec.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
     ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/my_matvec.pyx:1:6: Syntax error in simple statement list
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
my_matvec.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
     ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/my_matvec.pyx:1:6: Syntax error in simple statement list
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
my_matvec.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
     ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/my_matvec.pyx:1:6: Syntax error in simple statement list
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
my_matvec.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
     ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/my_matvec.pyx:1:6: Syntax error in simple statement list
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
my_matvec.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
maldun@hexenkessel:~/tex/hpc/docu/source$ cd ..
     ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/my_matvec.pyx:1:6: Syntax error in simple statement list
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
my_matvec.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
gcc -pthread -shared build/temp.linux-i686-2.6/my_matvec.o -L/home/maldun/sage/sage-4.6.1/local/lib -lpython2.6 -o /home/maldun/tex/hpc/docu/my_matvec.so
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include/numpy/__multiarray_api.h:1187: warning: ‘_import_array’ defined but not used
/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include/numpy/__ufunc_api.h:196: warning: ‘_import_umath’ defined but not used
gcc -pthread -shared build/temp.linux-i686-2.6/my_matvec.o -L/home/maldun/sage/sage-4.6.1/local/lib -lpython2.6 -o /home/maldun/tex/hpc/docu/my_matvec.so
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include/numpy/__multiarray_api.h:1187: warning: ‘_import_array’ defined but not used
/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include/numpy/__ufunc_api.h:196: warning: ‘_import_umath’ defined but not used
gcc -pthread -shared build/temp.linux-i686-2.6/my_matvec.o -L/home/maldun/sage/sage-4.6.1/local/lib -lpython2.6 -o /home/maldun/tex/hpc/docu/my_matvec.so
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include/numpy/__multiarray_api.h:1187: warning: ‘_import_array’ defined but not used
/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include/numpy/__ufunc_api.h:196: warning: ‘_import_umath’ defined but not used
gcc -pthread -shared build/temp.linux-i686-2.6/my_matvec.o -L/home/maldun/sage/sage-4.6.1/local/lib -lpython2.6 -o /home/maldun/tex/hpc/docu/my_matvec.so
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
  File "setup.py", line 9
    Extension("doubling", ["doubling.pyx"])]
            ^
SyntaxError: invalid syntax
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning my_matvec.pyx to my_matvec.c
building 'my_matvec' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c my_matvec.c -o build/temp.linux-i686-2.6/my_matvec.o
/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include/numpy/__multiarray_api.h:1187: warning: ‘_import_array’ defined but not used
/home/maldun/sage/sage-4.6.1/local/lib/python2.6/site-packages/numpy/core/include/numpy/__ufunc_api.h:196: warning: ‘_import_umath’ defined but not used
gcc -pthread -shared build/temp.linux-i686-2.6/my_matvec.o -L/home/maldun/sage/sage-4.6.1/local/lib -lpython2.6 -o /home/maldun/tex/hpc/docu/my_matvec.so
cythoning doubling.pyx to doubling.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
cdef doubling(double x):
    ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/doubling.pxd:1:5: function definition in pxd file must be declared 'cdef inline'

building 'doubling' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c doubling.c -o build/temp.linux-i686-2.6/doubling.o
doubling.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning doubling.pyx to doubling.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
cimport doubling

def call_doubling(double x):
    doubling.doubling(x)
           ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/doubling.pyx:4:12: Object of type 'object (double)' has no attribute 'doubling'
building 'doubling' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c doubling.c -o build/temp.linux-i686-2.6/doubling.o
doubling.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning doubling.pyx to doubling.c
building 'doubling' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c doubling.c -o build/temp.linux-i686-2.6/doubling.o
gcc -pthread -shared build/temp.linux-i686-2.6/doubling.o -L/home/maldun/sage/sage-4.6.1/local/lib -lpython2.6 -o /home/maldun/tex/hpc/docu/doubling.so
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning doubling.pyx to doubling.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
cimport doubling

def call_doubling(double x):
    return doubling.doubling(x)
                  ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/doubling.pyx:4:19: Object of type 'double (double)' has no attribute 'doubling'
building 'doubling' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c doubling.c -o build/temp.linux-i686-2.6/doubling.o
doubling.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning doubling.pyx to doubling.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
from doubling cimport doubling

def call_doubling(double x):
    return doubling.doubling(x)
                  ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/doubling.pyx:4:19: Object of type 'double (double)' has no attribute 'doubling'
building 'doubling' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c doubling.c -o build/temp.linux-i686-2.6/doubling.o
doubling.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu$ sage -python setup.py build_ext --inplace
running build_ext
cythoning doubling.pyx to doubling.c
building 'doubling' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c doubling.c -o build/temp.linux-i686-2.6/doubling.o
gcc -pthread -shared build/temp.linux-i686-2.6/doubling.o -L/home/maldun/sage/sage-4.6.1/local/lib -lpython2.6 -o /home/maldun/tex/hpc/docu/doubling.so
maldun@hexenkessel:~/tex/hpc/docu$ cd cython_tests/
maldun@hexenkessel:~/tex/hpc/docu/cython_tests$ sage -python setup.py build_ext --inplace
  File "setup.py", line 7
    cmdclass = {'build_ext': build_ext},
           ^
SyntaxError: invalid syntax
maldun@hexenkessel:~/tex/hpc/docu/cython_tests$ sage -python setup.py build_ext --inplace
running build_ext
cythoning math_stuff.pyx to math_stuff.c
building 'math_stuff' extension
creating build
creating build/temp.linux-i686-2.6
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c math_stuff.c -o build/temp.linux-i686-2.6/math_stuff.o
gcc -pthread -shared build/temp.linux-i686-2.6/math_stuff.o -L/home/maldun/sage/sage-4.6.1/local/lib -lm -lpython2.6 -o /home/maldun/tex/hpc/docu/cython_tests/math_stuff.so
maldun@hexenkessel:~/tex/hpc/docu/cython_tests$ sage -python setup.py build_ext --inplace
running build_ext
cythoning math_stuff.pyx to math_stuff.c
warning: /home/maldun/tex/hpc/docu/cython_tests/math_stuff.pyx:4:0: Overriding cdef method with def method.
building 'math_stuff' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c math_stuff.c -o build/temp.linux-i686-2.6/math_stuff.o
gcc -pthread -shared build/temp.linux-i686-2.6/math_stuff.o -L/home/maldun/sage/sage-4.6.1/local/lib -lm -lpython2.6 -o /home/maldun/tex/hpc/docu/cython_tests/math_stuff.so
maldun@hexenkessel:~/tex/hpc/docu/cython_tests$ sage -python setup.py build_ext --inplace
running build_ext
cythoning math_stuff.pyx to math_stuff.c
warning: /home/maldun/tex/hpc/docu/cython_tests/math_stuff.pyx:4:0: Overriding cdef method with def method.

Error converting Pyrex file to C:
------------------------------------------------------------
...
cdef extern from "math.h":
    double sin "c_sin" (double)

def sin(double x):
    return c_sin(x)
               ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/cython_tests/math_stuff.pyx:5:16: undeclared name not builtin: c_sin
building 'math_stuff' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c math_stuff.c -o build/temp.linux-i686-2.6/math_stuff.o
math_stuff.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu/cython_tests$ sage -python setup.py build_ext --inplace
running build_ext
cythoning math_stuff.pyx to math_stuff.c

Error converting Pyrex file to C:
------------------------------------------------------------
...
cdef extern from "math.h":
    double "c_sin" sin(double)
          ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/cython_tests/math_stuff.pyx:2:11: Empty declarator

Error converting Pyrex file to C:
------------------------------------------------------------
...
cdef extern from "math.h":
    double "c_sin" sin(double)
          ^
------------------------------------------------------------

/home/maldun/tex/hpc/docu/cython_tests/math_stuff.pyx:2:11: Syntax error in C variable declaration
building 'math_stuff' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c math_stuff.c -o build/temp.linux-i686-2.6/math_stuff.o
math_stuff.c:1:2: error: #error Do not use this file, it is the result of a failed Cython compilation.
error: command 'gcc' failed with exit status 1
maldun@hexenkessel:~/tex/hpc/docu/cython_tests$ sage -python setup.py build_ext --inplace
running build_ext
cythoning math_stuff.pyx to math_stuff.c
building 'math_stuff' extension
gcc -fno-strict-aliasing -g -O2 -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/home/maldun/sage/sage-4.6.1/local/include/python2.6 -c math_stuff.c -o build/temp.linux-i686-2.6/math_stuff.o
gcc -pthread -shared build/temp.linux-i686-2.6/math_stuff.o -L/home/maldun/sage/sage-4.6.1/local/lib -lm -lpython2.6 -o /home/maldun/tex/hpc/docu/cython_tests/math_stuff.so
maldun@hexenkessel:~/tex/hpc/docu/cython_tests$ 