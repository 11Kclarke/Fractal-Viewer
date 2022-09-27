"""base=["a","b","c","d"]
for i,val in reversed(list(enumerate(base))):
    print(i,val)
    if val=="b":
        print(base[i+1:])
"""
st="hello world   test          "
print(st[:2]+"!"+st[2:])