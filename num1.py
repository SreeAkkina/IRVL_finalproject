
print("Hi, your name! Welcome to intelligent Robotics and Vision Lab")
list1 = [10,7,30,15,26,3,85,45,77,37]
p=0
sum=0
min=11000000
for x in list1:
    if(x>50):
        print(x)
    if(p<x):
        p=x
    if(x<min):
        min=x
    
def sumoflist(list1, sum):
    for x in list1:
        sum+=x
    print(sum)

print(p)
print(min)
sumoflist(list1,sum)

