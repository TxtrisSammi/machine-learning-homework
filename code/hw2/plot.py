from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap as lcm

######################## QUESTION 1 #############################
x = np.linspace(-5 * np.pi, 5 * np.pi, 500)
f_x = x * (np.sin(x) ** 2)
g_x = -x * (np.sin(x) ** 2)

plt.title('Trigonometric Functions')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.plot(x, f_x, label='$f(x) = x\sin^2(x)$')
plt.plot(x, g_x, label='$g(x) = -x\sin^2(x)$')
plt.legend()
plt.show()
######################## QUESTION 2 #############################
# handling running in vscode or through terminal
try:
    file = open('./media/tcc.txt', 'r')
except:
    file = open('../../media/tcc.txt', 'r')
text = file.read().lower()

frequency = {}
for char in text:
    if char.isalpha() and char not in frequency:
        frequency[char] = 1
    elif char.isalpha():
        frequency[char] += 1

frequency = sorted(frequency.items(), key=lambda item: item[1])

letters, frequency = zip(*frequency)

plt.bar(letters, frequency)
plt.title('Character Frequency in "The Cosmic Computer" by Piper')
plt.xlabel('Alphabet')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
######################## QUESTION 3 #############################
board = np.arange(8*8).reshape(8, 8) % 2
# went with board[::2] instead of board[1::2] so letters are correct
board[1::2] = (board[::2] + 1) % 2 

# literally just cuz i wanna be special
colors = ["#26033d", "#d38f2a"] # [purple, brown]
colorMap = lcm(colors)

plt.imshow(board, cmap=colorMap)
plt.title('Chess Board Pattern')
plt.xticks(np.arange(8), list('ABCDEFGH'))
plt.yticks(np.arange(8), np.arange(8, 0, -1))
plt.tight_layout()
plt.show()
######################## QUESTION 3 #############################
a = np.matrix('1 0 1;2 1 1;0 1 1;1 1 2')
b = np.matrix('1 2 1;2 3 1;4 2 2')

ab = a * b

print(ab)
######################## QUESTION 3 #############################
n = np.arange(1,101)
f_n = np.cumsum((4*(-1)**(n+1))/((2*n)-1))
error = (np.pi - f_n) ** 2

plt.plot(f_n, label='Gregory Series')
plt.plot(error, label='Error')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.tight_layout()
plt.show()