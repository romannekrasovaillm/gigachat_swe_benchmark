#!/usr/bin/env python3

from sympy import symbols, apart

# Reproduce the exact issue
a = symbols('a', real=True)
t = symbols('t', real=True, negative=False)

bug = a * (-t + (-t + 1) * (2 * t - 1)) / (2 * t - 1)

print("Original expression:", bug)
print("After substituting a=1:")
result1 = bug.subs(a, 1)
print(result1)
print("Using apart():")
result2 = result1.apart()
print(result2)
print("Using apart(t):")
result3 = bug.apart(t)
print(result3)

# Check if they match
print("\nAre results equal?", result2.equals(result3))
