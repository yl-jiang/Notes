#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# evaltime_meta.py

from evalsupport import deco_alpha
from evalsupport import MetaAleph

print('<[1]> evaltime_meta module start')

@deco_alpha
class ClassThree:
    
    print('<[2]> ClassThree body')
    
    def method_y(self):
        print('<[3]> ClassThree.method_y')
        

class ClassFour(ClassThree):
    
    print('<[4]> ClassFour body')
    
    def method_y(self):
        print('<[5]> ClassFour.method_y')
        
class ClassFive(metaclass=MetaAleph):
 
    print('<[6]> ClassFive body')
    
    def __init__(self):
        print('<[7]> ClassFive.__init__')
        
    def method_z(self):
        print('<[8]> ClassFive.method_z')
        
class ClassSix(ClassFive):
    
    print('<[9]> ClassSix body')
    
    def method_z(self):
        print('<[10]> ClassSix.method_z')
        

if __name__ == '__main__':
    # 100， 400， 700， 1， 2， 200, 4， 6， 500, 9, 500
    print('<[11]> ClassThree tests', 30 * '.')  # 11
    three = ClassThree()
    three.method_y()  # 3
    print('<[12]> ClassFour tests', 30 * '.')  # 12
    four = ClassFour()  # ? 会不会去运行父类ClassThree的body
    four.method_y()  # 5
    print('<[13]> ClassFive tests', 30 * '.')  # 13
    five = ClassFive()  # 7, 500
    five.method_z()  # 600
    print('<[14]> ClassSix tests', 30 * '.')  # 14
    six = ClassSix()
    six.method_z()  # 10
    
print('<[15]> evaltime_meta module end')  # 15

