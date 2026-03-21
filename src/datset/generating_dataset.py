import sympy as smp
import random
import re

x = smp.symbols('x')

def random_poly(max_degree=6, coeff_range=(-9, 9)):
    
    """
    Generate random polynomial up to max_degree.
    Started with degree 4, but model needed more variety, so bumped to 6.
    Coefficient range -9 to 9 gives good spread without huge numbers.
    """
    
    deg = random.randint(1, max_degree)
    while True:
        expr = 0
        for d in range(deg + 1):
            c = random.randint(*coeff_range)
            if c != 0:
                expr += c * x**d
        expr = smp.expand(expr)
        
        if expr!=0 and expr.has(x):
            # Skip zero expressions or ones that simplify away x
            return smp.simplify(expr)

def random_analytic_function():

    """
    Generate random analytic function by combining: Polynomials, sin, cos, exp of polynomials,Adding another polynomial with 60% probability
    
    Tried adding division by quadratics (commented below) but caused 
    too many singularities in [-1,1], so disabled for now.
    """
    
    p1 = random_poly()
    p2 = random_poly()
    choices = [p1 , smp.sin(p1), smp.cos(p1), smp.exp(p1)]
    expr = random.choice(choices)
    if random.random() < 0.0:
        a = random.randint(-3, 3)
        b = random.randint(-3, 3)
        denom = 1 + a*x + b*x**2  
        expr = expr / denom
    if random.random() < 0.6:
        expr = expr + random_poly()
    return smp.expand(expr)

def taylor_series_up_to_fourth_order(expr):

    """
    SymPy's built-in series expansion up to O(x^5).
    Faster than manual differentiation, but kept manual version for verification.
    """
    
    return smp.series(expr, x, 0, 5).removeO()

def differentiation(f,x):
    return smp.diff(f,x)

def traditional_taylor_series_up_to_fourth_order(f):

    """
    Manual Taylor expansion using derivatives at 0.
    f(0) + f'(0)x + f''(0)x^2/2! + ...
    
    Kept this for verification — caught a few bugs in SymPy's series handling
    during early experiments. SymPy is fine now but verification stays.
    """
    
    res=f.subs(x,0)
    temp=f
    for i in range(1,5):
        temp=differentiation(temp,x)
        s=temp.subs(x,0)
        res=(res)+(s*(x**i)/smp.factorial(i))
    return res

TOKEN_RE = re.compile(
    r"""
    (?:\d+\.\d+|\d+)            
    |(?:[A-Za-z_]\w*)           
    |(?:\*\*|//|==|!=|<=|>=)    
    |(?:[+\-*/^()])             
    """,
    re.VERBOSE
)

def tokenize(expr_str: str):
    expr_str = expr_str.replace(" ", "")
    return TOKEN_RE.findall(expr_str)

import json
def generate_dataset(n_samples=2000, seed=42, out_path="taylor_tokenized_dataset.jsonl", verify=False):
    random.seed(seed)
    seen = set()
    with open(out_path, "w", encoding="utf-8") as f:
        kept = 0
        attempts = 0
        max_attempts = n_samples * 80   
        while kept < n_samples and attempts < max_attempts:
            attempts += 1
            if attempts % 20000 == 0:
                print("kept:", kept, "attempts:", attempts, "rate:", kept/attempts)
            try:
                fx = random_analytic_function()
                tx = taylor_series_up_to_fourth_order(fx)
                if verify:
                    tx2 = traditional_taylor_series_up_to_fourth_order(fx).expand()
                    if smp.simplify(tx - tx2) != 0:
                        continue
                fx_str = smp.sstr(fx)
                tx_str = smp.sstr(tx)
                if len(fx_str) > 120 or len(tx_str) > 160:
                    continue
                key = (fx_str, tx_str)
                if key in seen:
                    continue
                seen.add(key)
                item = {
                    "in": fx_str,
                    "out": tx_str,
                    "in_tokens": tokenize(fx_str),
                    "out_tokens": tokenize(tx_str),
                }
                f.write(json.dumps(item) + "\n")
                kept += 1
                if kept % 1000 == 0:
                    print(f"kept={kept}, attempts={attempts}")
            except Exception:
                continue
    print(f"Saved {kept} samples to {out_path} (attempts={attempts})")
generate_dataset(n_samples=31259, seed=7, out_path="taylor_tokenized_dataset.jsonl")
