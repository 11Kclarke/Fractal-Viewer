class test:
    b=1
    def __init__(self):
        self.a=1
        self.dic={"a":self.a}
    @classmethod
    def testfunc(self,a):
        a+=1
        
    def p(self):
        print(self.a)
        
        
T=test()
print(T.b)


T.testfunc(T.b)

print(T.b)
print(T.dic)
print(T.a)