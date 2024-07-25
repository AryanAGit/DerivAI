import sympy as sp
import random
import time
import re
start = time.time()
pairs = 0
numSet = 10000
#trig = 0, Poly = 1, Exp = 2, Log = 3, Arctrig = 4, const = 5, neg = 6
metaData = [0]*7
functionsCalled = [0] * 4
funLength = 15
#0 = composite funct, 1 = double composite funct, 2 = constant funct, 3 = single funct
temp = [0] * 4
numConst = int(numSet/12)
types = ""
#with open('function.txt', 'w') as file1, open('derivative.txt', 'w') as file2, open('functionToken.txt', 'w') as f3, open('dxToken.txt', 'w') as f4:
    #pass
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
    trig_functions = [sp.sin, sp.sec, sp.tan, sp.cos]
    trig_function = random.choice(trig_functions)
    coefficient = random.randint(1, 25)
    if random.randint(0, 1) == 1: 
        coefficient *= -1

    return coefficient * trig_function(variable)

def generate_random_arcTric():
    variable = sp.symbols('x')
    # Generate a random trigonometric function
    trig_functions = [sp.asin, sp.asec, sp.atan]
    trig_function = random.choice(trig_functions)
    coefficient = random.randint(1, 25)
    if random.randint(0, 1) == 1: 
        coefficient *= -1

    return coefficient * trig_function(variable)

def generate_random_exponential():
    variable = sp.symbols('x')
    # Generate a random exponential function
    coefficient = random.randint(1, 25)
    if random.randint(0, 1) == 1: 
        coefficient *= -1
    base = random.randint(2, 15)

    return sp.Mul( coefficient, sp.exp(variable * sp.log(base)))

def generate_random_log():
    variable = sp.symbols('x')
    # Generate a random exponential function
    coefficient = random.randint(1, 25)
    if random.randint(0, 1) == 1: 
        coefficient *= -1
    
    base = random.randint(2, 6)

    return sp.Mul( coefficient, sp.log(variable, base))

def generate_random_polynomial():
    variable = sp.symbols('x')
    degree = random.randint(1, 7)
    if random.randint(0, 2) == 0:
        coefficient = random.randint(1, 7)
        if random.randint(0, 1) == 1: 
            coefficient *= -1
        return coefficient * variable ** degree
    coefficients = [random.randint(-10, 10) for _ in range(degree)]
    for i in range(len(coefficients)):
        rand = random.randint(0,2)
        if rand == 0 and degree != 1:
            coefficients[i] = 0
    func = sum(c * variable**i for i, c in enumerate(coefficients))
    return func

def genNegPower():
    variable = sp.symbols('x')
    degree = random.randint(1, 7)
    if random.randint(0, 2) == 0:
        coefficient = random.randint(1, 7)
        if random.randint(0, 1) == 1: 
            coefficient *= -1
        return coefficient / variable ** degree
    coefficients = [random.randint(-10, 10) for _ in range(degree)]
    for i in range(len(coefficients)):
        rand = random.randint(0,3)
        if rand == 0 and degree != 1:
            coefficients[i] = 0
    func = sum(c / variable**(-1*i) for i, c in enumerate(coefficients))
    return func

def randF():
    f = [generate_random_trigonometric, generate_random_polynomial, generate_random_exponential, generate_random_log, generate_random_arcTric, genNegPower]

    index = random.randint(0, 5)
    reroll = random.randint(0, 20)
    if reroll < 6:
        index = 1
    elif reroll <10:
        index = 2
    elif reroll <12:
        index = 3
    elif reroll < 14:
        index = 4
    elif reroll < 17:
        index = 0
    elif reroll < 19:
        index = 5

    func = f[index]()

    if intReject(str(func)):
        global types
        if index == 1:
            types += "poly"
        elif index == 5:
            types += "neg"
    
        return func
        
    else:
        return randF()

def composite(f):
    g = randF()
    num = random.randint(0, 3)
    if num == 0:
        return g - f
    elif num == 1:
        return g+f
    elif num == 2:
        return f*g
    elif num == 3: 
        return f/g
 
def generate_function_and_derivative():
    global types
    x = sp.symbols('x')
    choice = random.randint(0, 25)
    if choice < 12 or choice > 23:
        function = composite(randF())
        derivative = sp.diff(function, x)
        temp[0] += 1
        return function, derivative
    elif choice < 16:
        function = randF() * randF()
        derivative = sp.diff(function, x)
        temp[1] += 1
        return function, derivative
    elif choice < 18 and metaData[5] < numConst: 
        function = sp.Mul(1, random.randint(-100, 100))
        derivative = sp.diff(function, x)
        temp[2] += 1
         
        types += "const"
        return function, derivative
    else:
        function = randF()
        derivative = sp.diff(function, x)
        temp[3] += 1
        return function, derivative
    
def classify(f, data):
    if ("sin" in f and "asin" not in f) or ("cos" in f) or ("tan" in f and "atan" not in f) or ("sec" in f and "asec" not in f):
        data[0] += 1
    if "exp" in f:
        data[2] += 1
       
    if "log" in f and "exp" not in f:
        data[3] += 1

    if "asin" in f or "asec" in f or "atan" in f:
        data[4] += 1
  
    if "neg" in f:
        data[6] += 1
    if "poly" in f:
        data[1] += 1
    if "const" in f:
        data[5] += 1
def classify2(f):
    types = ""
    if ("sin" in f and "asin" not in f) or ("cos" in f) or ("tan" in f and "atan" not in f) or ("sec" in f and "asec" not in f):
        types += "trig"
    if "exp" in f:
        types += "exp"
       
    if "log" in f and "exp" not in f:
        types += "log"

    if "asin" in f or "asec" in f or "atan" in f:
        types += "arc"
  
    if "neg" in f:
        types += "neg"
    if "poly" in f:
        types += "poly"
    if "const" in f:
        types += "const"
    if types == "":
        return "unknown"
    return types
with open('testingF.txt', 'a') as file1, open('testingD.txt', 'a') as file2, open ('testingTypes.txt', 'a') as file3:
    while(pairs < numSet):

        func, dx = generate_function_and_derivative()
        valid = True
        f = str(func)
        if "zoo" in f or "nan" in f or f == "0":
            valid = False
        f += "\n"
        d = str(dx) + "\n"
        tokenD = tokenize(d)
        tokenF = tokenize(f)
        if len(tokenD) < funLength and len(tokenF) < funLength and valid:
          

            for token in tokenF:
                file1.write(token + " ")
            file1.write("\n")

            for token in tokenD:
                file2.write(token + " ")
            file2.write("\n")
            pairs += 1
            classify(types+f, metaData)
            types = classify2(types + f)
            file3.write(types + "\n")
            for i in range(4):
                functionsCalled[i] += temp[i]
         
        temp [:] = [0]*4
        types = ""
        
print("Trig, Poly, exp, log, arctrig, const, neg")    
print(metaData)
print("\nComp func*func Const Rand")
print(functionsCalled)
elapsed = time.time()-start
print(f"Elapsed time: {elapsed} seconds\n\n")


