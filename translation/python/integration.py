import math

n = 1000
a = -math.pi
b = 2*math.pi/3
dx = (b - a) / n
sum = 0.0

for i in range(n):
    x = a + i * dx
    sum += math.sin(x) * dx

print("The area between -π and 2/3π for sin(x) is: ", sum)

exact_result = -math.cos(b) + math.cos(a)
print("The exact area between -π and 2/3π for sin(x) is: ", exact_result)
