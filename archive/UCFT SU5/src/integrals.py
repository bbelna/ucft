import sys
import sympy as sp

# Increase recursion limit to help with deep expansions
sys.setrecursionlimit(10000)

# Initialize pretty printing
sp.init_printing(use_unicode=True)

# -------------------------------
# Define symbols and parameters
# -------------------------------
epsilon, M2, mu2 = sp.symbols('epsilon M2 mu2', positive=True, real=True)
mu = sp.sqrt(mu2)  # optional if you want a sqrt
d = 4 - epsilon  # d = 4 - epsilon

# We define gamma_E as EulerGamma
gammaE = sp.EulerGamma
I_S_expr = None

# -------------------------------
# 1. One-Loop Integral: Direct Encoded Formula
#    I_S(M^2) = - M^4 / (2(4π)^2) [ 2/ε + 1 - γ_E + ln(4πμ^2 / M^2 ) ] + const
#    We store it as an expression in terms of M2 and epsilon.
# -------------------------------
def one_loop_integral_encoded(M2, epsilon, mu2):
    """
    Directly encode the known result for the one-loop scalar integral
    I_S(M^2), avoiding Sympy's indefinite integration to prevent stalling.
    The standard expression in dimensional regularization is:

      I_S(M^2) = - (M^4)/(2(4π)^2) * [
           2/ε + 1 - γ_E + ln(4π μ^2 / M^2)
      ] + const

    We'll treat the constant as 0 for convenience.
    """
    # We'll define:
    # M^4 = (M2^2)
    M4 = M2**2
    # The prefactor
    prefactor = -M4 / (2*(4*sp.pi)**2)
    # The bracket
    bracket = (2/epsilon) + 1 - gammaE + sp.log(4*sp.pi*mu2/M2)
    I_S_expr = prefactor * bracket
    return sp.simplify(I_S_expr)

def one_loop_effective_potential_encoded(M2, epsilon, mu2):
    """
    Delta V = 1/2 I_S(M^2)
    """
    I_S_expr = one_loop_integral_encoded(M2, epsilon, mu2)
    return sp.Rational(1,2)*I_S_expr

# -------------------------------
# 2. Two-Loop and Three-Loop: we keep them as before
# -------------------------------
def sunset_integral(M2, epsilon, mu2):
    d = 4 - epsilon
    I_sunset = sp.gamma(3 - d) / (2 * (4*sp.pi)**(d)) * (M2)**(d - 3)
    return sp.simplify(I_sunset)

def basketball_integral(M2, epsilon, mu2):
    d = 4 - epsilon
    I_basketball = sp.gamma(4 - 3*d/sp.Integer(2)) / (2 * (4*sp.pi)**(3*d/sp.Integer(2)) * sp.gamma(4)) * (M2)**(3*d/sp.Integer(2) - 4)
    return sp.simplify(I_basketball)

def double_bubble_integral(M2, epsilon, mu2):
    """
    Encoded known dimensional regularization formula:
    Double-bubble integral = [I_S(M^2)]^2
    """
    I_S = one_loop_integral_encoded(M2, epsilon, mu2)
    I_double = I_S**2
    return sp.simplify(I_double)

# -------------------------------
# Helper: Safe Series Expansion
# -------------------------------
def safe_series(expr, var, x0, n):
    try:
        expr_simpl = sp.simplify(expr)
        series_expr = sp.series(expr_simpl, var, x0, n)
    except Exception as e:
        print(f"Series expansion failed: {e}")
        series_expr = sp.latex(expr)
    return series_expr

# -------------------------------
# Write LaTeX Output to File
# -------------------------------
def write_latex_output(results_dict, filename="./results.tex"):
    with open(filename, "w") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage{amsmath, amssymb}" + "\n")
        f.write(r"\begin{document}" + "\n")
        for title, expr in results_dict.items():
            f.write(r"\section*{" + title + "}" + "\n")
            f.write(r"\begin{align*}" + "\n")
            f.write(expr + "\n")
            f.write(r"\end{align*}" + "\n\n")
        f.write(r"\end{document}" + "\n")
    print(f"LaTeX output written to {filename}")

# -------------------------------
# Main Routine
# -------------------------------
def main():
    print("BEGIN")

    # 1. One-loop integral (encoded formula)
    I_S_expr = one_loop_integral_encoded(M2, epsilon, mu2)
    print("One-loop integral (encoded) expression done.")
    I_S_series = safe_series(I_S_expr, epsilon, 0, 2)
    print("One-loop series expansion completed.")

    # One-loop effective potential
    DeltaV_expr = one_loop_effective_potential_encoded(M2, epsilon, mu2)
    print("One-loop effective potential (encoded) done.")
    DeltaV_series = safe_series(DeltaV_expr, epsilon, 0, 2)
    print("One-loop effective potential series expansion completed.")

    # 2. Two-loop integrals
    I_double = double_bubble_integral(M2, epsilon, mu2)
    print("Double-bubble diagram computed.")
    I_double_series = safe_series(I_double, epsilon, 0, 2)
    print("Double-bubble series expansion computed.")
    
    I_sunset_expr = sunset_integral(M2, epsilon, mu2)
    print("Sunset diagram computed.")
    I_sunset_series = safe_series(I_sunset_expr, epsilon, 0, 2)
    print("Sunset series expansion computed.")

    # 3. Three-loop integral
    I_basket_expr = basketball_integral(M2, epsilon, mu2)
    print("Basketball diagram computed.")
    I_basket_series = safe_series(I_basket_expr, epsilon, 0, 2)
    print("Basketball series expansion computed.")
    
    # 4. Beta function structure summary (as LaTeX)
    beta_eq = sp.latex(sp.Eq(sp.symbols(r"\beta(\lambda)"),
                      -sp.symbols("c_1")*sp.symbols("lambda")**2 
                      + sp.symbols("c_2")*sp.symbols("lambda")**3 
                      - sp.symbols("c_3")*sp.symbols("lambda")**4 + sp.symbols(r"\cdots")))

    # Collect results
    results = {
        "One-loop Scalar Integral $I_S$ (series in $\epsilon$)": sp.latex(I_S_series),
        "One-loop Effective Potential $\Delta V$ (series in $\epsilon$)": sp.latex(DeltaV_series),
        "Double-Bubble Diagram (two-loop, series in $\epsilon$)": sp.latex(I_double_series),
        "Sunset Diagram (two-loop, series in $\epsilon$)": sp.latex(I_sunset_series),
        "Basketball Diagram (three-loop, series in $\epsilon$)": sp.latex(I_basket_series),
        "Beta Function Structure Summary": beta_eq
    }
    
    try:
        write_latex_output(results)
    except Exception as e:
        print("Error writing LaTeX file:", e)

if __name__ == "__main__":
    main()
