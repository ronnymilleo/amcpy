from functions import gmax, mean, std, meanofsquared

print("Tests")

n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x = gmax(n)
y = mean(n)
z = std(n)
a = meanofsquared(n)

print(x)
print(y)
print(z)
print(a)
