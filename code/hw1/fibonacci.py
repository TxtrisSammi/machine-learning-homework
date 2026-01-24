def fibonacci(n):
    # implement me
    firstFibNum, secondFibNum = 0, 1
    for i in range(n):
        yield firstFibNum
        firstFibNum, secondFibNum = secondFibNum, firstFibNum + secondFibNum

for t in fibonacci(50):
    print(t)