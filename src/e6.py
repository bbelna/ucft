#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E6 structure constants and coset expansion script in 6D.
This version enumerates the 72 roots by closure under addition, starting
from the simple roots.  The simple roots here are chosen consistently with
the standard E6 Cartan matrix in e6_cartan_matrix().
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, Tuple, List

import sympy
from sympy import Matrix, sqrt, symbols, simplify, nsimplify, Integer

###############################################################################
# GLOBAL CONFIGURATION
###############################################################################
DOT_TOL = 1e-9
JACOBI_RANDOM_TESTS = 500

###############################################################################
# ARGUMENT PARSER
###############################################################################
parser = argparse.ArgumentParser(
    description="Compute and validate E6 structure constants in 6D via direct enumeration."
)
parser.add_argument("--cartan-only", action="store_true")
parser.add_argument("--expand-maurer", action="store_true")
parser.add_argument("--validate", action="store_true")
parser.add_argument("--full-jacobi", action="store_true")
parser.add_argument("--to-file", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

###############################################################################
# LOGGING CONFIGURATION
###############################################################################
if args.to_file:
    if os.path.exists("e6.log"):
        os.remove("e6.log")
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(filename="e6.log", filemode="w", level=level,
                        format="%(asctime)s %(levelname)s: %(message)s")
else:
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

###############################################################################
# DATA CLASS
###############################################################################
@dataclass
class RootData:
    alpha: Matrix
    label: str

###############################################################################
# HELPER: CANONICAL KEY FOR SYMBOLIC VECTORS
###############################################################################
def canonical_key(v: Matrix) -> Tuple:
    # Use nsimplify so that small algebraic expressions unify
    return tuple(nsimplify(x) for x in v)

###############################################################################
# 1) CARTAN MATRIX
###############################################################################
def e6_cartan_matrix():
    return Matrix([
        [2, -1,  0,  0,  0,  0],
        [-1, 2, -1,  0,  0,  0],
        [0, -1,  2, -1,  0,  0],
        [0,  0, -1,  2, -1, -1],
        [0,  0,  0, -1,  2,  0],
        [0,  0,  0, -1,  0,  2]
    ])

###############################################################################
# 2) DEFINE SIMPLE ROOTS IN 6D
###############################################################################
def define_simple_roots_e6():
    from sympy import Matrix

    e1 = Matrix([1,0,0,0,0,0])
    e2 = Matrix([0,1,0,0,0,0])
    e3 = Matrix([0,0,1,0,0,0])
    e4 = Matrix([0,0,0,1,0,0])
    e5 = Matrix([0,0,0,0,1,0])
    e6 = Matrix([0,0,0,0,0,1])

    a1 = e1 - e2
    a2 = e2 - e3
    a3 = e3 - e4
    a4 = e4 - e5
    a5 = e5 - e6

    # The "magic" alpha6:
    #   (-2/3, -2/3, -2/3, -2/3, 1/3, 1/3)
    # Check: norm^2 = 4*(4/9) + 2*(1/9) = 2
    #        alpha4·alpha6 = (0,0,0,1,-1,0)·(...) = -1
    a6 = Matrix([-sympy.Rational(2,3), -sympy.Rational(2,3),
                 -sympy.Rational(2,3), -sympy.Rational(2,3),
                  sympy.Rational(1,3),  sympy.Rational(1,3)])
    
    return [a1, a2, a3, a4, a5, a6]

###############################################################################
# OPTIONAL: CONSISTENCY CHECK
###############################################################################
def check_simple_roots_against_cartan(simple_roots, C):
    """
    Quick sanity check: for each i,j, verify that
        2 * (alpha_i dot alpha_j) / (alpha_j dot alpha_j) == C[i,j]
    up to small numerical tolerance.
    """
    import math
    n = len(simple_roots)
    for i in range(n):
        for j in range(n):
            lhs = 2.0 * float(simple_roots[i].dot(simple_roots[j])) / float(simple_roots[j].dot(simple_roots[j]))
            rhs = float(C[i,j])
            if abs(lhs - rhs) > 1e-7:
                logging.warning(
                    f"Mismatch Cartan[{i+1},{j+1}]: "
                    f"2(a{i+1}·a{j+1})/(a{j+1}·a{j+1})={lhs:.3f}, but C={rhs:.3f}."
                )

###############################################################################
# 3) GENERATE POSITIVE ROOTS BY CLOSURE UNDER ADDITION
###############################################################################
def generate_e6_positive_roots(simple_roots: List[Matrix]) -> List[Matrix]:
    """
    Starting from the simple roots, repeatedly add pairs of positive roots.
    If the sum has squared norm exactly 2, add it to the set.
    Continue until no new positive roots appear.  For a correct set of
    simple roots matching E6, this should yield exactly 36 positive roots.
    """
    R: Dict[Tuple, Matrix] = {}
    # Initialize R with the simple roots.
    for r in simple_roots:
        key = canonical_key(r)
        R[key] = simplify(r)
    iteration = 0
    changed = True
    while changed:
        iteration += 1
        changed = False
        current = list(R.values())
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                candidate = simplify(current[i] + current[j])
                if candidate == 0:
                    continue
                # Check squared length
                length_sq = candidate.dot(candidate).evalf()
                # Because these are all real exact expressions, we can test ==2 symbolically:
                if abs(length_sq.evalf() - 2) < 1e-12:
                    key = canonical_key(candidate)
                    if key not in R:
                        R[key] = simplify(candidate)
                        changed = True
        logging.info(f"Direct closure iteration {iteration}: {len(R)} positive roots so far.")
    return list(R.values())

###############################################################################
# 4) STRUCTURE CONSTANTS AND RELATED FUNCTIONS (SYMPY–BASED)
###############################################################################
def build_cartan_generators(simple_roots: List[Matrix]) -> List[Matrix]:
    # Typically, the Cartan subalgebra can be identified with these
    return simple_roots

def init_E_map(all_roots: List[Matrix]) -> Dict[Tuple, str]:
    E_map = {}
    idx = 1
    for r in all_roots:
        key = canonical_key(r)
        if key not in E_map:
            E_map[key] = f"E_{idx}"
            idx += 1
    return E_map

def compute_brackets(H_list: List[Matrix],
                     all_roots: List[Matrix],
                     E_map: Dict[Tuple, str],
                     tol=1e-9):
    bracket_dict = {}
    def kf(v: Matrix):
        return canonical_key(v)

    # H-H brackets all vanish in a standard Lie algebra basis
    for i in range(len(H_list)):
        for j in range(i+1, len(H_list)):
            bracket_dict[(("H", kf(H_list[i])), ("H", kf(H_list[j])))] = []

    # H-E bracket => alpha(H)*E_alpha
    def alpha_action(a: Matrix, H: Matrix):
        return simplify(a.dot(H))

    for i, H in enumerate(H_list):
        Hk = kf(H)
        for r in all_roots:
            rk = kf(r)
            if rk not in E_map:
                continue
            val = alpha_action(r, H)
            # if val != 0 => [H, E_r] = val * E_r
            if val != 0 and abs(float(val.evalf())) > tol:
                bracket_dict[(("H", Hk), ("E", E_map[rk]))] = [(val, ("E", E_map[rk]))]

    # E-E bracket => either 0, or another E_{r1+r2} if r1+r2 is a root,
    # or belongs to the Cartan if r1 + r2 = 0
    root_set = set(kf(r) for r in all_roots)
    for r1 in all_roots:
        r1k = kf(r1)
        if r1k not in E_map:
            continue
        E1 = E_map[r1k]
        for r2 in all_roots:
            r2k = kf(r2)
            if r2k not in E_map or r1k == r2k:
                continue
            E2 = E_map[r2k]
            sum_vec = simplify(r1 + r2)
            length_sq = sum_vec.dot(sum_vec)
            # If r1 + r2 = 0 => [E_{r1}, E_{r2}] is in Cartan
            if length_sq == 0:
                # sum up the "alpha(H)" piece
                # i.e. [E_r, E_{-r}] ~ H_r
                comm_terms = []
                for h_idx, H in enumerate(H_list):
                    Hk = kf(H)
                    c = alpha_action(r1, H)
                    if c != 0 and abs(float(c.evalf())) > tol:
                        comm_terms.append((c, ("H", Hk)))
                if comm_terms:
                    bracket_dict.setdefault((("E", E1), ("E", E2)), []).extend(comm_terms)
            else:
                # If r1 + r2 is a root => bracket is E_{r1+r2}, up to a sign
                sumk = kf(sum_vec)
                if sumk in root_set:
                    # typical convention: [E_r, E_s] = +/- E_{r+s}
                    # the sign is root‐system dependent, but we usually pick
                    # a convention so that if r < s, you get plus sign, else minus
                    sgn = Integer(1) if (r1k < r2k) else Integer(-1)
                    bracket_dict.setdefault((("E", E1), ("E", E2)), []).append((sgn, ("E", E_map[sumk])))

    # Enforce antisymmetry
    bracket_dict = enforce_antisymmetry(bracket_dict)
    return bracket_dict

def enforce_antisymmetry(bracket_dict):
    new_keys = list(bracket_dict.keys())
    for (X, Y) in new_keys:
        rev = (Y, X)
        if rev not in bracket_dict:
            terms = bracket_dict[(X, Y)]
            bracket_dict[rev] = [(-c, op) for (c, op) in terms]
    return bracket_dict

def identify_coset_generators(root_data, H_U1, tol=DOT_TOL, expected=32):
    coset = []
    for rd in root_data:
        dot_val = simplify(rd.alpha.dot(H_U1))
        if abs(float(dot_val.evalf())) > tol:
            coset.append(("E", rd.label))
    if len(coset) != expected:
        logging.error(f"Identified {len(coset)} coset generators, but expected {expected}!")
        return None
    return coset

def init_coset_symbols(coset_ops: List[Tuple[str, str]]):
    n = len(coset_ops)
    pi_syms = symbols(f"pi1:{n+1}", real=True)
    dpi_syms = symbols(f"dpi1:{n+1}", real=True)
    X_terms = []
    dX_terms = []
    from sympy import I
    for i, lab in enumerate(coset_ops):
        X_terms.append((I*pi_syms[i], lab))
        dX_terms.append((I*dpi_syms[i], lab))
    return X_terms, dX_terms, pi_syms, dpi_syms

def commutator_of_sums(A_terms, B_terms, bracket_dict):
    out = []
    for (a_cf, A_op) in A_terms:
        for (b_cf, B_op) in B_terms:
            pairs = bracket_dict.get((A_op, B_op), [])
            for (c, Z_op) in pairs:
                out.append((a_cf * b_cf * c, Z_op))
    return out

def combine_like_terms(terms):
    from sympy import simplify
    cdict = {}
    for (coef, op) in terms:
        cdict[op] = cdict.get(op, 0) + coef
    out = []
    for op, val in cdict.items():
        simp_val = simplify(val)
        if simp_val.is_number:
            if abs(float(simp_val.evalf())) > 1e-14:
                out.append((simp_val, op))
        else:
            out.append((simp_val, op))
    return out

def expand_maurer_cartan(X_terms, dX_terms, bracket_dict):
    from sympy import Integer
    comm = commutator_of_sums(X_terms, X_terms, bracket_dict)
    half_comm = [(c/Integer(2), op) for (c, op) in comm]
    Omega = dX_terms + half_comm
    return combine_like_terms(Omega)

def validate_brackets(bracket_dict, H_list, all_roots, E_map, full_jacobi=False):
    import random, sympy
    logging.info("Validation: checking [H,H], [H,E], antisymmetry, Jacobi.")
    def kf(v: Matrix):
        return canonical_key(v)

    # 1) [H_i, H_j] = 0
    for i in range(len(H_list)):
        for j in range(i+1, len(H_list)):
            comm = bracket_dict.get((("H", kf(H_list[i])), ("H", kf(H_list[j]))), [])
            if comm:
                logging.error(f"[H_{i},H_{j}] != 0 => {comm}")
                return False

    # 2) [H_i, E_alpha] = alpha(H_i) E_alpha
    for r in all_roots:
        rk = kf(r)
        if rk not in E_map:
            continue
        E_label = E_map[rk]
        for i, H in enumerate(H_list):
            expected = sympy.simplify(H.dot(r))
            comm = bracket_dict.get((("H", kf(H)), ("E", E_label)), [])
            if abs(float(expected.evalf())) < 1e-12:
                # expect 0
                s = 0
                for (c, _) in comm:
                    s += abs(float(sympy.simplify(c).evalf()))
                if s > 1e-12:
                    logging.error(f"[H_{i},E_{E_label}] nonzero but expected 0 => {comm}")
                    return False
            else:
                # expect exactly 1 term => (expected, E_label)
                if len(comm) != 1:
                    logging.error(f"[H_{i},E_{E_label}] has {len(comm)} terms => {comm}")
                    return False
                c = comm[0][0]
                diff = float(sympy.simplify(c - expected).evalf())
                if abs(diff) > 1e-12:
                    logging.error(f"[H_{i},E_{E_label}] mismatch => got {c}, want {expected}")
                    return False

    # 3) Antisymmetry check: [X,Y] = -[Y,X]
    keys = list(bracket_dict.keys())
    sample = min(200, len(keys))
    for key in random.sample(keys, sample):
        rev = (key[1], key[0])
        t1 = bracket_dict.get(key, [])
        t2 = bracket_dict.get(rev, [])
        cdict = {}
        for (cf, op) in t1:
            cdict[op] = cdict.get(op, 0) + cf
        for (cf, op) in t2:
            cdict[op] = cdict.get(op, 0) + cf
        for val in cdict.values():
            if abs(float(sympy.simplify(val).evalf())) > 1e-12:
                logging.error(f"Antisymmetry fails for {key}")
                return False

    # 4) Jacobi identity check
    ops = set()
    for (X, Y) in bracket_dict.keys():
        ops.add(X)
        ops.add(Y)
    ops = list(ops)

    def bracket_sum(opA, terms):
        out = []
        for (cf, opB) in terms:
            t = bracket_dict.get((opA, opB), [])
            for (cf2, opC) in t:
                out.append((cf*cf2, opC))
        return out

    def combine(terms):
        from sympy import simplify
        cdict = {}
        for (cf, op) in terms:
            cdict[op] = cdict.get(op, 0) + cf
        out = []
        for op, val in cdict.items():
            val_simpl = simplify(val)
            if abs(float(val_simpl.evalf())) > 1e-12:
                out.append((val_simpl, op))
        return out

    def jacobi_test(X, Y, Z):
        # [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
        tYZ = bracket_dict.get((Y,Z), [])
        tZX = bracket_dict.get((Z,X), [])
        tXY = bracket_dict.get((X,Y), [])

        tot = []
        tot.extend(bracket_sum(X, tYZ))
        tot.extend(bracket_sum(Y, tZX))
        tot.extend(bracket_sum(Z, tXY))

        combined = combine(tot)
        return (len(combined) == 0)

    nTests = (len(ops)**3 if full_jacobi else 200)
    for _ in range(nTests):
        X = random.choice(ops)
        Y = random.choice(ops)
        Z = random.choice(ops)
        # skip trivial cases
        if X == Y or Y == Z or Z == X:
            continue
        if not jacobi_test(X, Y, Z):
            logging.error(f"Jacobi fails for {X}, {Y}, {Z}")
            return False

    logging.info("Validation passed.")
    return True

###############################################################################
# MAIN
###############################################################################
def main():
    overall_start = time.perf_counter()
    logging.info("=== E6 Structure Constants and Coset Expansion (6D) ===")
    
    # 1. Cartan Matrix
    C = e6_cartan_matrix()
    logging.info(f"Cartan matrix for E6:\n{C}")

    # 2. Define simple roots (Sympy)
    t0 = time.perf_counter()
    sr_sympy = define_simple_roots_e6()

    # Optional check they match the Cartan matrix
    check_simple_roots_against_cartan(sr_sympy, C)

    t1 = time.perf_counter()
    logging.info(f"Defined {len(sr_sympy)} simple roots (Sympy) in {t1 - t0:.4f} seconds.")
    for i, r in enumerate(sr_sympy, 1):
        logging.info(f"a_{i} = {r}")

    if args.cartan_only:
        sys.exit(0)

    # 3. Generate positive roots via closure.
    t2 = time.perf_counter()
    positive_roots = generate_e6_positive_roots(sr_sympy)
    t3 = time.perf_counter()
    logging.info(f"Direct closure generated {len(positive_roots)} positive roots in {t3 - t2:.4f} seconds.")
    if len(positive_roots) != 36:
        logging.error(f"Expected 36 positive roots, found {len(positive_roots)}! Aborting.")
        return
    all_roots = positive_roots + [simplify(-r) for r in positive_roots]
    if len(all_roots) != 72:
        logging.error(f"Expected 72 total roots, found {len(all_roots)}! Aborting.")
        return
    else:
        logging.info(f"Found exactly {len(all_roots)} roots. Good.")

    # 4. Define H_U1 and compute brackets
    # (example: sometimes used in supergravity or model building contexts)
    H_U1 = (3/sympy.sqrt(30)) * Matrix([1,1,1,1,1,-5])
    logging.info(f"Using H_U1 = {H_U1}")

    t4 = time.perf_counter()
    H_list = build_cartan_generators(sr_sympy)
    E_map = init_E_map(all_roots)
    logging.info("Constructing bracket dictionary...")
    bracket_dict = compute_brackets(H_list, all_roots, E_map, tol=1e-9)
    t5 = time.perf_counter()
    logging.info(f"Bracket dictionary has {len(bracket_dict)} entries (computed in {t5 - t4:.4f} seconds).")

    # 5. Optionally, expand the Maurer-Cartan form
    if args.expand_maurer:
        logging.info("Identifying coset generators (expect 32)...")
        root_data = []
        for r in all_roots:
            k = canonical_key(r)
            if k in E_map:
                root_data.append(RootData(alpha=r, label=E_map[k]))
        coset_ops = identify_coset_generators(root_data, H_U1, DOT_TOL, expected=32)
        if coset_ops is None:
            logging.error("Coset identification dimension mismatch. Aborting.")
            return
        logging.info(f"Identified {len(coset_ops)} coset generators. Good.")
        X_terms, dX_terms, pi_syms, dpi_syms = init_coset_symbols(coset_ops)
        Omega = expand_maurer_cartan(X_terms, dX_terms, bracket_dict)
        logging.info("Maurer-Cartan expansion: Ω = dX + 1/2[X,X] =")
        for (cf, op) in Omega:
            logging.info(f"  + ({cf}) * {op}")

    # 6. Optionally, validate the brackets
    if args.validate:
        logging.info("Running validations...")
        t6 = time.perf_counter()
        ok = validate_brackets(bracket_dict, sr_sympy, all_roots, E_map, full_jacobi=args.full_jacobi)
        t7 = time.perf_counter()
        if not ok:
            logging.error("Validation FAILED.")
        else:
            logging.info(f"Validation PASSED (took {t7 - t6:.4f} seconds).")

    overall_end = time.perf_counter()
    logging.info(
        f"=== E6 Structure Constants and Coset Expansion: Done in {overall_end - overall_start:.4f} seconds ==="
    )

if __name__ == "__main__":
    main()
