
def euclidean_dist(v1,v2):
    if len(v1)!=len(v2):
        print("Vector sizes must have same dimensions.")
    result=0
    for i in range(len(v1)):
        result += ((v1[i]-v2[i])**2)
    return result**(0.5)

def manhattan_dist(v1,v2):
    if len(v1)!=len(v2):
        print("Vector sizes must have same dimensions.")
    result=0
    for i in range(len(v1)):
        result += ((v1[i]-v2[i]))
    return abs(result)

def main():
    print("Enter the 1st vector:")
    v1=input()
    d1=v1.split(',')
    v1=[]
    for a in d1:
        v1.append(int(a))
    print("Enter the 2nd vector:")
    v2=input()
    d2=v2.split(',')
    v2=[]
    for a in d2:
        v2.append(int(a))
    print("Euclidean distance:",euclidean_dist(v1,v2))
    print("Manhattan distance:",manhattan_dist(v1,v2))
    
main()