ar=[]
n=int(input("enter number of elements:"))
print("enter the elements")
for i in range(0,n):
	x=int(input())
	ar.append(x)
print(ar)
for i in range(0,n):
	for j in range(i+1,n):
		if ar[i]>ar[j]:
			temp=ar[i]
			ar[i]=ar[j]
			ar[j]=temp
print("the sorted array are :",ar)

"""
OUTPUT

enter number of elements:5
enter the elements
10
5
12
8
1
[10, 5, 12, 8, 1]
the sorted array are : [1, 5, 8, 10, 12]"""


