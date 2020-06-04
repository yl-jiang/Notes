#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from evalsupport import deco_alpha

print('<[1]> evaltime module start')

class ClassOne:
    
    print('<[2]> ClassOne body')
    
    def __init__(self):
        print('<[3]> ClassOne.__init__')
        
    def __del__(self):
        print('<[4]> ClassOne.__del__')
        
    def method_x(self):
        print('<[5]> ClassOne.method_x')
        
    class ClassTwo(object):
        print('<[6]> ClassTwo body')
        
    
@deco_alpha
class ClassThree:
    
    print('<[7]> ClassThree body')
    
    def method_y(self):
        print('<[8]> ClassThree.method_y')
        
    
class ClassFour(ClassThree):
    
    print('<[9]> ClassFour body')
    
    def method_y(self):
        print('<[10]> ClassFour.method_y')
        

if __name__ == "__main__":
    print('<[11]> ClassOne tests', 30 * '.')  # 11
    one = ClassOne()  # 3
    one.method_x()  # 5
    print('<[12]> ClassThree tests', 30 * '.')  # 12
    three = ClassThree()
    three.method_y()  # 300
    print('<[13]> ClassFour tests', 30 * '.')  # 13
    four = ClassFour()
    four.method_y()  # 10
    
print('<[14]> evaltime module end')  # 14

