import sympy as sp
import random
import time
import re
start = time.time()
pairs = 0
numSet = 10000
#trig = 0, Poly = 1, Exp = 2, Log = 3, Arctrig = 4, const = 5
data = [0]*6
metaData = [0]*6
with open('function.txt', 'w') as file1, open('derivative.txt', 'w') as file2, open('functionToken.txt', 'w') as f3, open('dxToken.txt', 'w') as f4:
    pass
def tokenize(f):
    token_pattern = re.compile(r'\d|\w+|[^\s\w]')
    tokens = token_pattern.findall(f)
    return tokens

def intReject(s):
    try:
        int(s)
        return False
    except ValueError:
        return True

def generate_random_trigonometric():
    variable = sp.symbols('x')
    # Generate a random trigonometric function
    trig_functions = [sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc, 
                      sp.sinh, sp.cosh, sp.tanh, sp.coth, sp.sech, sp.csch]
    trig_function = random.choice(trig_functions)
    coefficient = random.randint(-10, 10)

    return coefficient * trig_function(variable)

def generate_random_arcTric():
    variable = sp.symbols('x')
    # Generate a random trigonometric function
    trig_functions = [sp.asin, sp.acos, sp.atan, sp.asinh, sp.acosh, sp.atanh]
    trig_function = random.choice(trig_functions)
    coefficient = random.randint(-10, 10)

    return coefficient * trig_function(variable)

def generate_random_exponential():
    variable = sp.symbols('x')
    # Generate a random exponential function
    coefficient = random.randint(-10, 10)
    base = random.randint(2, 5)

    return sp.Mul( coefficient, sp.exp(variable * sp.log(base)))

def generate_random_log():
    variable = sp.symbols('x')
    # Generate a random exponential function
    coefficient = random.randint(-10, 10)
    base = random.randint(2, 5)

    return sp.Mul( coefficient, sp.log(variable, base))

def generate_random_polynomial():
    variable = sp.symbols('x')
    coefficients = [random.randint(-10, 10) for _ in range(random.randint(1, 5))]
    for i in range(len(coefficients)):
        rand = random.randint(0,2)
        if rand == 0:
            coefficients[i] = 0
    return sum(c * variable**i for i, c in enumerate(coefficients))

def randF():
    f = [generate_random_trigonometric, generate_random_polynomial, generate_random_exponential, generate_random_log, generate_random_arcTric]
    index = random.randint(0, 4)
    if random.randint(0, 2) == 0:
        index = 1
    func = f[index]()
    if intReject(str(func)):
        data[index] += 1
        return func
    else:
        return randF()
def composite(f):
    g = randF()
    return f.subs(sp.symbols('x'), g)

def generate_function_and_derivative():
    data [:] = [0]*6
    x = sp.symbols('x')
    choice = random.randint(0, 20)
    if choice < 4:
        function = composite(randF())
        derivative = sp.diff(function, x)
        return function, derivative
    elif choice < 6:
        function = composite(composite(randF()))
        derivative = sp.diff(function, x)
        return function, derivative
    elif choice == 6: 
        function = sp.Mul(1, random.randint(-10, 10))
        derivative = sp.diff(function, x)
        data[5] += 1
        return function, derivative
    else:
        function = randF()
        derivative = sp.diff(function, x)
        return function, derivative

# Generate a dataset

with open('function.txt', 'a') as file1, open('derivative.txt', 'a') as file2, open('functionToken.txt', 'a') as file3, open('dxToken.txt', 'a') as file4:
    while(pairs < numSet):
        func, dx = generate_function_and_derivative()
        f = str(func) + "\n"
        d = str(dx) + "\n"
        if len(f) < 50 and len(d) < 50:
            file1.write(f)
            file2.write(d)
            tokenF = tokenize(f)
            for token in tokenF:
                file3.write(token + " ")
            file3.write("\n")
            tokenD = tokenize(d)
            for token in tokenD:
                file4.write(token + " ")
            file4.write("\n")
            pairs += 1
            for i in range(6):
                metaData[i] += data[i]

print("Trig, Poly, exp, log, arctrig")    
print(metaData)
elapsed = time.time()-start
print(f"Elapsed time: {elapsed} seconds\n\n")


for i in range(4):
    f , _ = generate_function_and_derivative()

    print(str(f))
    print("\n")
    t = tokenize(str(f))
    for token in t:
        print(token)
    print("\n\n")
    
