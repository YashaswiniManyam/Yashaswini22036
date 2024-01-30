def common(a,b):
    set_a=set(a)
    set_b=set(b)
    
    if len(set_a.intersection(set_b))>0:
        return(set_a.intersection(set_b))

a=input('')
a = {a for a in a.split(",")}
b=input('')
b = {b for b in b.split(",")}

count=0
for i in common(a,b):
    count+=1
print(count)