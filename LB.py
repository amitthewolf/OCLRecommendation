number = [1,2,3]
print([x-10 for x in number if x==2])

def betza(x):
    def mashu(y):
        return x + y
    return mashu

res = betza(5)
print(res(6))

def add(x, y):
    return x+y


print(list(map(add, number,number)))

def palundrom(text):
    return text[::2]

print(palundrom("aba"))