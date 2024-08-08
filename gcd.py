def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Input from the user
num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))

# Calculate the GCD
result = gcd(num1, num2)

# Print the result
print(f"The GCD of {num1} and {num2} is {result}")

'''
OUTPUT

Enter the first number: 50
Enter the second number: 60
The GCD of 50 and 60 is 10
'''
