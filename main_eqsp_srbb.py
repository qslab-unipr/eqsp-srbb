import os
import sys
import math
import time
import numpy as np
from eqsp_utils import (
    wrap_to_pi,
    generateThetaVect,
    b_from_theta_level,
    solve_Mk_recursive,
    build_Mk_explicit,
    b_from_state_phases_mean,
    check_level_solution_explicit,   # or check_level_solution_subset
    )
import pennylane as qml
from qsp_srbb_circuit import test_srbb_method_params


def bitstring(i: int, n: int) -> str:
    return format(i, f"0{n}b")

def print_state_components(unit_vector: np.ndarray, n: int, *, atol=1e-14):
    print("\n=== TARGET STATE COMPONENTS ===")
    for i, amp in enumerate(unit_vector):
        r = abs(amp)
        phi = np.angle(amp)
        if r < atol:
            phi = 0.0
        print(f"|{bitstring(i,n)}>  amp={amp.real:+.6e}{amp.imag:+.6e}j   |amp|={r:.6e}   phase={phi:+.6f} rad")

def format_state_components(psi, label="state", kmax=None, tol=1e-12, sort_by_prob=True):
    """
    Stampa componenti dello stato:
      idx | amp (re,im) | |amp| | phase(rad) | prob
    - kmax: mostra solo i primi kmax (dopo ordinamento se sort_by_prob=True)
    """
    psi = np.asarray(psi, dtype=complex)
    probs = np.abs(psi)**2
    mags = np.abs(psi)
    phases = np.angle(psi)
    phases = np.array([wrap_to_pi(p) for p in phases])

    idxs = np.arange(len(psi))
    if sort_by_prob:
        idxs = idxs[np.argsort(-probs)]

    if kmax is not None:
        idxs = idxs[:kmax]

    print(f"\n=== {label}: components (showing {len(idxs)} of {len(psi)}) ===")
    print(" idx |    amp(re)        amp(im)    |    |amp|      phase(rad)     prob")
    print("-"*78)
    for i in idxs:
        if probs[i] < tol and mags[i] < tol:
            continue
        a = psi[i]
        print(f"{i:>4} | {a.real:+.6e}  {a.imag:+.6e} | {mags[i]:.6e}  {phases[i]:+.6f}  {probs[i]:.6e}")

def print_theta_grouped(theta_vector: np.ndarray, N: int):
    print("\n=== THETA VECTOR FROM BST (FULL) ===")
    print(theta_vector)

    print("\n=== THETA VECTOR GROUPED BY LEVEL ===")
    print(f"level 0: theta0 = {theta_vector[0]:+.3e}  (single Ry)")
    start = 1
    for l in range(1, N):
        count = 2**l
        block = theta_vector[start:start+count]
        print(f"level {l}: thetas[{start}:{start+count-1}] (len={count}) = {block}")
        start += count

def print_M_pretty(M: np.ndarray):
    # stampa righe come + e -
    for row in M:
        s = "".join("+" if v > 0 else "-" for v in row)
        print(s)

def print_qsp_circuit(qnode, paramsMod, paramsPhase, state=True, initial_state=None, title=None):
    """
    Visualizza il circuito QSP-SRBB con i parametri correnti.
    state=True -> ritorna statevector
    """
    if title is not None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    drawer = qml.draw(qnode)
    print(
        drawer(
            paramsMod,
            paramsPhase,
            state,
            initial_state,
            False   # dm=False
        )
    )

def compare_states_mod_phase(psi_t, psi_o, label="comparison", kmax=16, tol=1e-12):
    psi_t = np.asarray(psi_t, dtype=complex)
    psi_o = np.asarray(psi_o, dtype=complex)

    mag_t, mag_o = np.abs(psi_t), np.abs(psi_o)
    ph_t, ph_o = np.angle(psi_t), np.angle(psi_o)

    # allinea fase globale usando overlap
    overlap = np.vdot(psi_t, psi_o)
    g = np.angle(overlap)
    psi_o_aligned = psi_o * np.exp(-1j*g)

    mag_oa = np.abs(psi_o_aligned)
    ph_oa = np.angle(psi_o_aligned)

    probs_t = mag_t**2
    idxs = np.argsort(-probs_t)[:kmax]

    print(f"\n=== {label}: target vs output (global-phase aligned) ===")
    print(" idx |     |t|          |o|         Δ| |      ph(t)    ph(o_al)  Δph(wrap)   prob(t)")
    print("-"*92)
    for i in idxs:
        if probs_t[i] < tol and mag_t[i] < tol:
            continue
        dmag = mag_oa[i] - mag_t[i]
        dph  = wrap_to_pi(ph_oa[i] - ph_t[i])
        print(f"{i:>4} | {mag_t[i]:.6e} {mag_oa[i]:.6e} {dmag:+.2e}  {ph_t[i]:+.6f} {ph_oa[i]:+.6f} {dph:+.3e}  {probs_t[i]:.6e}")

    # metriche rapide sul modulo e sulla fase
    print("\nQuick checks:")
    print(f"max |Δ|amp|| = {np.max(np.abs(mag_oa - mag_t)):.3e}")
    # fase ha senso solo dove il modulo è non piccolo
    mask = mag_t > 1e-10
    if np.any(mask):
        dph_all = np.array([wrap_to_pi(x) for x in (ph_oa[mask] - ph_t[mask])])
        print(f"phase RMS (masked) = {np.sqrt(np.mean(dph_all**2)):.3e}")
    else:
        print("phase RMS (masked) = n/a (all magnitudes ~0)")

def phase_align(psi_t, psi_o):
    g = np.angle(np.vdot(psi_t, psi_o))
    return psi_o * np.exp(-1j*g)


np.set_printoptions(precision=3, suppress=True, linewidth=120)

result_path = 'results/'
if not os.path.exists(result_path):
    os.makedirs(result_path)


DEBUG_STATE = True          # print coeff + modulus + phase for each component
DEBUG_THETA = True          # print theta_vector (full and by level)
DEBUG_LEVELS = True         # print b, x, residues for each level
DEBUG_PRINT_M = False        # print M_k
DEBUG_PRETTY_M = True       # M_k as rows of +/-
PRINT_M_UP_TO_K = 5         # for N=4, k ranges up to 4
ATOL_CHECK = 1e-9
RTOL_CHECK = 1e-9


try:
    match input("\nSpecific quantum state [s] or random vector [r]? > ").strip().lower():
        case 's':
            match int(input("Select a known quantum state or insert a custom one:\n" \
            "1) Bell Phi+ (N=2)\n" \
            "2) Bell Phi- (N=2)\n" \
            "3) Bell Psi+ (N=2)\n" \
            "4) Bell Psi- (N=2)\n" \
            "5) GHZ (N=3)\n" \
            "6) GHZ (N=4)\n" \
            "7) W (N=3)\n" \
            "8) W (N=4)\n" \
            "9) Dicke (N=3)\n" \
            "10) Dicke (N=4)\n" \
            "11) Custom\n" \
            " > ").strip()):
                case 1:
                    N = 2
                    coefficients = [math.sqrt(1/2), 0, 0, math.sqrt(1/2)]
                case 2:
                    N = 2
                    coefficients = [math.sqrt(1/2), 0, 0, -math.sqrt(1/2)]
                case 3:
                    N = 2
                    coefficients = [0, math.sqrt(1/2), math.sqrt(1/2), 0]
                case 4:
                    N = 2
                    coefficients = [0, math.sqrt(1/2), -math.sqrt(1/2), 0]
                case 5:
                    N = 3
                    coefficients = [math.sqrt(1/2), 0, 0, 0, 0, 0, 0, math.sqrt(1/2)]
                case 6:
                    N = 4
                    coefficients = [math.sqrt(1/2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, math.sqrt(1/2)]
                case 7:
                    N = 3
                    coefficients = [0, math.sqrt(1/3), math.sqrt(1/3), 0, math.sqrt(1/3), 0, 0, 0]
                case 8:
                    N = 4
                    coefficients = [0, math.sqrt(1/4), math.sqrt(1/4), 0, math.sqrt(1/4), 0, 0, 0, math.sqrt(1/4), 0, 0, 0, 0, 0, 0, 0]
                case 9:
                    N = 3
                    coefficients = [0, 0, 0, math.sqrt(1/3), 0, math.sqrt(1/3), math.sqrt(1/3), 0]
                case 10:
                    N = 4
                    coefficients = [0, 0, 0, math.sqrt(1/6), 0, math.sqrt(1/6), math.sqrt(1/6), 0, 0, math.sqrt(1/6), math.sqrt(1/6), 0, math.sqrt(1/6), 0, 0, 0]
                case 11:
                    N = int(input("Number of qubits? > ").strip())
                    coefficients = list(map(float, input("Insert real coefficients (floats) separated by spaces > ").strip().split()))
                case _:
                    raise Exception("Invalid input!")

        case 'r':
            N = int(input("Number of qubits? > ").strip())

            match input("Dense [d] or sparse [s] random state? > ").strip().lower():
                case 'd':
                    match input("Real positive [p], real negative [n] or complex [c] coefficients? > ").strip().lower():
                        case 'p':
                            numbers = np.abs(np.random.normal(size=2**N))
                        case 'n':
                            numbers = np.random.normal(size=2**N)
                        case 'c':
                            numbers = np.random.normal(size=2**N) + 1j *  np.random.normal(size=2**N)
                        case _:
                            raise Exception("Invalid input!")
                        
                    numbers /= np.linalg.norm(numbers, ord=2)
                    coefficients = numbers.tolist()
                    
                case 's':
                    match input("Real positive [p], real negative [n] or complex [c] coefficients? > ").strip().lower():
                        case 'p':
                            numbers = np.abs(np.random.normal(size=2**N))
                        case 'n':
                            numbers = np.random.normal(size=2**N)
                        case 'c':
                            numbers = np.random.normal(size=2**N) + 1j *  np.random.normal(size=2**N)
                        case _:
                            raise Exception("Invalid input!")
                    
                    zero_indices = np.random.choice(2**N, np.random.randint(1, 2**N), replace=False)
                    numbers[zero_indices] = 0
                    numbers /= np.linalg.norm(numbers, ord=2)
                    coefficients = numbers.tolist()

                case _:
                    raise Exception("Invalid input!")
        case _:
            raise Exception("Invalid input!")
        
except Exception as e:
    print(e)
    sys.exit(1)

#sys.stdout = open(result_path + 'qsp_output.txt', 'w', encoding='utf-8') # Comment this to write on console (comment also stdout.close() at the end)


unit_vector = np.array(coefficients, dtype=complex)
assert np.isclose(np.linalg.norm(unit_vector), 1.0, atol=1e-12), "State not normalized"
if DEBUG_STATE:
    print_state_components(unit_vector, N)


##### START GENERATION/VERIFICATION OF PARAMETERS #####

param_start_time = time.perf_counter()

## ---- Classical precomputation of parameters (moduli ladder) ---- ##

theta_vector = generateThetaVect(unit_vector, N)   # BST angles (len = 2^N - 1)
if DEBUG_THETA:
    print_theta_grouped(theta_vector, N)

print("\n=== CLASSICAL PRECOMPUTATION: UCG levels for moduli ===")

if DEBUG_LEVELS:
    print("\nPer-level linear systems: M_k @ x = b")

x_levels = []
for l in range(1, N):
    if DEBUG_LEVELS:
        print(f"\n--- Level l={l}  (UCG on {l+1} qubits) ---")

    # 1) build RHS b^{(l)}
    b = b_from_theta_level(theta_vector, l)

    # 2) solve M_{l+1} @ x = b
    x = solve_Mk_recursive(b)
    
    if DEBUG_LEVELS:
        print(f"b (len={len(b)}): {b}")
        print(f"x (len={len(x)}): {x}")

    # 3) verification
    ok, max_abs, resid = check_level_solution_explicit(b, x, atol=1e-9, rtol=1e-9)
    
    if DEBUG_LEVELS:
        print(f"check ok={ok}, max|resid|={max_abs:.9e}")
        if not ok:
            print("residual vector (M_k @ x - b):")
            print(resid)
    
    # Convention fix: PennyLane RZ(2θ) implements diag(e^{-iθ}, e^{+iθ}),
    # while SRBB linear systems assume diag(e^{+iθ}, e^{-iθ}). Flip sign to match.
    x = -x
    x_levels.append(np.asarray(x, dtype=float))

    # print M_k (only if requested and k not too much large)
    k = l + 1  # since len(b)=2^(l+1)=2^k
    if DEBUG_PRINT_M and (k <= PRINT_M_UP_TO_K):
        M = build_Mk_explicit(k)
        print(f"M_{k} shape = {M.shape}")
        if DEBUG_PRETTY_M:
            print_M_pretty(M)
        else:
            print(M)


## ---- Classical precomputation of parameters (complex phase Z-block) ---- ##

print("\n=== CLASSICAL PRECOMPUTATION: max level for phases ===\n")

b_phase, g0, alpha_total = b_from_state_phases_mean(unit_vector)

x_phase = solve_Mk_recursive(b_phase)

if DEBUG_LEVELS:
        print(f"b_phase (len={len(b_phase)}): {b_phase} -> must correspond to b1")
        print(f"x_phase (len={len(x_phase)}): {x_phase}")

ok_p, max_abs_p, resid_p = check_level_solution_explicit(b_phase, x_phase, atol=ATOL_CHECK, rtol=RTOL_CHECK)

if DEBUG_LEVELS:
        print(f"check ok={ok_p}, max|resid|={max_abs_p:.9e}")
        if not ok_p:
            print("residual vector (M_k @ x - b):")
            print(resid_p)

# Convention fix: PennyLane RZ(2θ) implements diag(e^{-iθ}, e^{+iθ}),
# while SRBB linear systems assume diag(e^{+iθ}, e^{-iθ}). Flip sign to match.
x_phase = -x_phase

# print M_k (only if requested and k not too much large)
k = l + 1  # since len(b)=2^(l+1)=2^k
if DEBUG_PRINT_M and (k <= PRINT_M_UP_TO_K):
    M = build_Mk_explicit(k)
    print(f"M_{k} shape = {M.shape}")
    if DEBUG_PRETTY_M:
        print_M_pretty(M)
    else:
        print(M)


param_end_time = time.perf_counter()

##### END GENERATION/VERIFICATION OF PARAMETERS #####


print("\n=== QUANTUM STAGE: reordering and mapping parameters + run SRBB-QSP circuit ===")

# MODULO: paramsMod = [gamma0] + concat_{n=2..N}(x_level_n)
# PHASE: paramsPhase = x_phase (len = 2^N - 1)

# paramsMod contiene i parametri per la preparazione dei moduli:
#   - gamma0: angolo RY iniziale sul primo qubit
#   - x_levels: soluzioni dei sistemi lineari per ciascun livello n=2..N di UCG
# L'ordine è importante ma il riordino fisico è gestito dal circuito SRBB.

# paramsPhase contiene i parametri per la preparazione delle fasi:
# è il vettore soluzione del sistema lineare per il blocco Z massimo (n=N),
# con scelta di fase globale media.

gamma0 = float(2*theta_vector[0])
theta_mod_concat = np.concatenate(x_levels) if x_levels else np.array([], dtype=float)
paramsMod = np.concatenate(([gamma0], theta_mod_concat))
paramsPhase = np.asarray(x_phase, dtype=float)

print(f"N={N}")
print(f"paramsMod: len={len(paramsMod)} (gamma0 + sum levels)")
print(f"paramsPhase: len={len(paramsPhase)} (max Z-block)")

# Esecuzione del circuito EQSP-SRBB:
# - il primo blocco prepara i moduli (QSP ladder with SRBB insertions)
# - poi applica il blocco diagonale massimo per le fasi complesse
# Il risultato è lo stato |psi_out>.


dev = qml.device("default.qubit", wires=N)
qnode = test_srbb_method_params(dev, N)

#print_qsp_circuit(
#    qnode,
#    paramsMod,
#    paramsPhase,
#    state=True,
#    initial_state=None,
#    title="EQSP-SRBB circuit (MODULO + PHASE)"
#)

psi_out = qnode(paramsMod, paramsPhase, True, initial_state=None, dm=False)

# Fidelity (invariante per fase globale)
# Fidelity: F = |<psi_target | psi_out>|^2
overlap = np.vdot(unit_vector, psi_out)
fidelity = float(np.abs(overlap) ** 2)

# Trace distance per stati puri: sqrt(1 - F)
trace_distance = float(np.sqrt(max(0.0, 1.0 - fidelity)))

print("\n=== OUTPUT CHECK ===")
print(f"overlap <psi_target|psi_out> = {overlap.real:+.6e}{overlap.imag:+.6e}j")
print(f"fidelity = {fidelity:.16f}")
print(f"trace distance (pure-state) = {trace_distance:.16e}")


#print("dtype psi_out:", np.asarray(psi_out).dtype)
#print("dtype overlap:", np.asarray(overlap).dtype)
#print("fidelity repr:", repr(fidelity))
#print("1 - fidelity:", (1.0 - fidelity))
#print(f"fidelity (sci) = {fidelity:.18e}")
#print(f"1-fidelity (sci) = {(1.0-fidelity):.18e}")


#fidelity_raw = float(np.abs(overlap)**2)
#fidelity = min(1.0, max(0.0, fidelity_raw))
#trace_distance = float(np.sqrt(max(0.0, 1.0 - fidelity)))

#print(f"fidelity_raw = {fidelity_raw:.18e}")
#print(f"fidelity_clamped = {fidelity:.18e}")
#print(f"1-fidelity_clamped = {(1.0-fidelity):.18e}")


if fidelity < 1 - 1e-8:
    print("\n[WARN] Fidelity is not ~1. Consider checking:\n"
          "  - phase convention (mean global phase)\n"
          "  - b_phase construction\n"
          "  - mapping/order functions in qsp_srbb_circuit.py\n"
          "  - numerical tolerances")

# Stampa dettagliata delle componenti dello stato target/output:
# - modulo delle ampiezze
# - fase associata

#format_state_components(unit_vector, label="TARGET", kmax=16, sort_by_prob=True)
#format_state_components(psi_out, label="OUTPUT", kmax=16, sort_by_prob=True)
#compare_states_mod_phase(unit_vector, psi_out, label="TARGET vs OUTPUT", kmax=16)

# Allineamento di fase globale:
# rimuove una fase globale arbitraria prima di confrontare le componenti.

psi_out_al = phase_align(unit_vector, psi_out)
l2 = np.linalg.norm(unit_vector - psi_out_al)
linf = np.max(np.abs(unit_vector - psi_out_al))
print(f"phase-aligned L2 error = {l2:.3e}")
print(f"phase-aligned Linf error = {linf:.3e}")

# Metriche basate sulle probabilità |psi|^2:
# - TVD: Total Variation Distance
# - Hellinger^2: distanza geometrica tra distribuzioni
# entrambe invarianti per fase.

p = np.abs(unit_vector)**2
q = np.abs(psi_out)**2
tvd = 0.5*np.sum(np.abs(p-q))
print(f"TVD(probs) = {tvd:.3e}")

hell2 = 1.0 - np.sum(np.sqrt(p*q))
hell2 = max(0.0, float(hell2))
print(f"Hellinger^2(probs) = {hell2:.3e}")

# Errore di fase RMS pesato:
# confronta le fasi solo dove l'ampiezza target è non trascurabile,
# pesando per la probabilità.

psi_out_al = phase_align(unit_vector, psi_out)
ph_t = np.angle(unit_vector)
ph_o = np.angle(psi_out_al)
mask = np.abs(unit_vector) > 1e-10
dph = np.array([wrap_to_pi(x) for x in (ph_o[mask] - ph_t[mask])])
w = (np.abs(unit_vector)[mask]**2)
phase_mse = np.sum(w * dph**2) / np.sum(w)
print(f"weighted phase RMS = {np.sqrt(phase_mse):.3e}")