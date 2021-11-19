# Style Guide For Writing Python Code for Rumi

### PEP guidelines

Stick to PEP guidelines as much as possible
https://www.python.org/dev/peps/pep-0008/

### Documentation
- Every function exposed to the user must be documented with docstrings.
- Docstring is first line in any function definition.
- Example is shown below. Observe the empty lines in docstring. 

```
def my_function(arg1, arg2, arg3="something"):
	"""Here goes short description
	
	Here goes long description of function
	
	Parameters
	----------
	arg1: str
	    description for arg1
	arg2: int
	    description for arg2
	arg3: str, optional
	    description for arg3
		
		
	Returns
	-------
	list
	   description for what it returns..returns a list of strings
	"""
	
	return ["_".join([arg1, arg3]) for i in range(arg2)]

```
- Similar structure can be followed for modules
```
"""This is first line, and short description for module

And here is long description
"""
```


### Naming conventions
1. Variables and functions should all be small case. in case the name 
has multiple words, the words should be joined with `_`. e.g.

```
def hello_world(name):
    print("Hello",name)
	
	
valid_variable

def valid_function():
	pass
```

2. Classes should follow camel case. class, instance variables and methods follow
same guideline as for variable and functions as described above.

### Coding style
1. Code is more often read than written. So give nice names to variables and
functions which convey business logic.
2. Functions should have optimal number of lines. A big function is difficult 
to read. A function which can't be read in single screen needs rewrite by
refactoring it into smaller functions.
3. Division of function into smaller parts in not only by logic of length,
but also by steps involved. A function to generate prime numbers can be 
written as single complex function. But nice division into smaller 
steps makes it easy to understand and code!

```
def primes(n):
	p = []
    for i in range(n):
        i_is_prime = True
	    for j in range(2,i):
            i_is_prime = i_is_prime and i%j==0
		if i_is_prime:
		    p.append(i)
	return p
```

this can be refactored into a much readable code.

```
def factors(n):
    return [f for f in range(n) if n%f==0]
	
def is_prime(p):
    return factors(p) == [1,p]
	
def primes(n):
    return [p for p in range(n) if is_prime(n)]
```
