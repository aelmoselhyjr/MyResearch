#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:27:19 2022

@author: alielmoselhy
"""

# class produce
# class variable

class Variable():
    def __init__(self, name):
        self.name=name
        self.var = True
class Constant():
    def __init(self, value):
        self.value = value
        self.var = False
class Product():
    def __init__(self, term1, term2):
        self.term1 = term1
        self.term2 = term2
        self.var = bool(term1.var + term2.var)
class Exponent():
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
        self.var = bool(base.var + exponent.var)
class Sum():
    def __init__(self, term1, term2):
        self.term1 = term1
        self.term2 = term2

#def deriv(expression, var):
    