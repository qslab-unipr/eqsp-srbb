from math import acos
import numpy as np

"""
def binaryInnProd(x : int, y : int, n : int) -> str: # Compute the inner product in F^2 between two states of 'n' qubits
    v = []
    for i in range(n):
        v.append(int(x[i]) * int(y[i]))

    return bin(sum(v))[-1]

def compBasis(n: int) -> str: # Generate the computational basis vectors for 'n' qubits
    strings = []

    for k in range(pow(2, n)):
        strings.append(bin(k)[2:].zfill(n))

    return strings
"""

def cartesian_to_polar(vector):
    magnitudes = np.abs(vector)
    angles = np.angle(vector)
    return magnitudes, angles

"""
def toKet(vector : list, n : int) -> str: # Show the KET notation of an input vector of 'n' qubits
    basis = compBasis(n)
    state = ''

    for i, q in enumerate(vector):
        if q != 0:
            state += (str(q) + '|' + str(basis[i]) + '>  ')

    return state

def grayCode(n_gray : int, n : int):
    up_prefix = np.array([[0]])
    down_prefix = np.array([[1]])
    matrix = np.concatenate((up_prefix, down_prefix))

    match n_gray:
        case 1:
            for _ in range(n - 1):
                up_prefix = np.concatenate((up_prefix, up_prefix))
                down_prefix = np.concatenate((down_prefix, down_prefix))
                matrix = np.concatenate((np.concatenate((matrix, np.flip(matrix, 0))), np.concatenate((up_prefix, down_prefix))), axis = 1)
        case 2:
            for _ in range(n - 1):
                up_prefix = np.concatenate((up_prefix, up_prefix))
                down_prefix = np.concatenate((down_prefix, down_prefix))
                matrix = np.concatenate((np.concatenate((up_prefix, down_prefix)), np.concatenate((matrix, np.flip(matrix, 0)))), axis = 1)
            
    return matrix

def stateToVector(state : np.ndarray):
    state_vector = 1
    for ks in state:
        vector = np.array([1 - ks, ks]) # Vector notation of ket state
        state_vector = np.kron(state_vector, vector)
    return state_vector

def circuitMatrixReduction(diag_matrix, n, m):
    dim_full = 2 ** (n + m)
    selected_indices = [i for i in range(dim_full) if i & ((1 << m) - 1) == 0]
    reduced_diag = np.diag([diag_matrix[i, i] for i in selected_indices])
    return reduced_diag
"""

class Node:
    def __init__(self, value, arc):
        self._value = value # Value of the node
        self._arc_val = arc # Value of the arc from node to parent
        self._children = [] # Children nodes

    def addChild(self, node: 'Node'): # Add a brach to the tree
        self._children.append(node)
    
    def nodeVal(self) -> float: # Value of the node
        return self._value
    
    def arcVal(self) -> str: # Value of the arc
        return self._arc_val
    
    def explore(self, string : str) -> float: # Explore recursively the tree until the desire state
        if not string:
            return self.nodeVal()
        
        token = string[0]
        for childe in self._children:
            if childe.arcVal() == token:
                return childe.explore(string[1:])
    
    def printTree(self, depth : int = 0):
        for childe in self._children:
            childe.printTree(depth + 1)
        ind = '-' * depth
        print(ind + str(self.nodeVal()))

def generateBinTree(vector_v : list, bit : str, k : int): # Generate a binary search tree of depth 'k' from a list of coeffiecients
    value = np.linalg.norm(vector_v, ord = 2) # Norm-2
    node = Node(value, bit)

    if k != 0:
        vector0 = vector_v[:len(vector_v)//2]
        vector1 = vector_v[len(vector_v)//2:]

        node.addChild(generateBinTree(vector0, '0', k - 1))
        node.addChild(generateBinTree(vector1, '1', k - 1))

    return node

def generateThetaVect(unit_vector : np.ndarray, n : int): # Generate the vector of theta from a multiplexor rappresentation in binary strings
    theta = np.array([])

    binary_strings = encodeMultiplexor(np.array(['0']), n)
    binary_strings = sorted(binary_strings, key=len)

    bst = generateBinTree(unit_vector, None, n)

    for child_string in binary_strings:
        
        parent_string = child_string[:-1]
        child = bst.explore(child_string)
        parent = bst.explore(parent_string)

        if parent == 0:
            theta = np.append(theta, acos(0))
        else:
            theta = np.append(theta, acos(child / parent))

    return theta

"""
def generateInnProdMatrix(n: int) -> np.ndarray: # Generate the matrix of inner products between every computational basis vectors except |00...00>
    basis_vectors = compBasis(n)[1:]
    coefficients = np.array([])

    for i in basis_vectors:
        row = np.array([])
        for j in basis_vectors:
            row = np.append(row, float(binaryInnProd(i, j, n)))
        coefficients = np.append(coefficients, row)
    coefficients = np.reshape(coefficients, (len(basis_vectors), len(basis_vectors)))

    return coefficients

def generateAlphaVect(coefficients : np.ndarray, theta_vector : np.ndarray) -> np.ndarray: # Generate the vector of alpha by solving the linear system
    alpha_vector = np.linalg.solve(coefficients, theta_vector)
    return alpha_vector
"""

def encodeMultiplexor(binary_vector : np.ndarray, n : int) -> np.ndarray: # Generate all the binary strings to use for the exploration of the binary tree
    n = n - 1

    if n == 0:
        return binary_vector
    
    elem_0 = '0' + binary_vector[-1]
    branch_0 = np.append(binary_vector, elem_0)
    branch_0 = encodeMultiplexor(branch_0, n)

    elem_1 = '1' + binary_vector[-1]
    branch_1 = np.append(binary_vector, elem_1)
    branch_1 = encodeMultiplexor(branch_1, n)

    binary_vector = np.union1d(branch_0, branch_1)

    return binary_vector

"""
def matrixLambdaFinal(state_vector):

    # Step 1: Print the desired quantum state
    #print("\n[Step 1] Input quantum state:\n")
    #print(state_vector)
    #assert np.isclose(np.linalg.norm(state_vector), 1, atol = 1e-6), f"Coefficient vector norm is not 1: {np.linalg.norm(state_vector)} != 1."

    # Step 2: Cartesian coordinates (real and imaginary parts)
    #real_parts = np.real(state_vector)
    #imag_parts = np.imag(state_vector)

    #print("\n[Step 2] Cartesian coordinates for each component:\n")
    #print("Real part: ", real_parts)
    #print("Immaginary part: ", imag_parts)
 
    # Step 3: Cartesian to polar
    magnitudes, angles = cartesian_to_polar(state_vector)
    #print("\n[Step 3] Polar coordinates for each component:\n")
    #print("Magnitudes: ", magnitudes)
    #print("Phase angles [rad]: ", angles)

    # Step 4: Magnitudes and complex phases
    magnitudes = np.abs(state_vector)
    phases = np.array([z / abs(z) if abs(z) > 1e-10 else 1.0 for z in state_vector])
    #print("\n[Step 4] Euler representation for each component:\n")
    #print("Magnitudes: ", magnitudes)
    #print("Phases: ", phases)

    # Step 5: Collect the first entry in the array "phases"
    phase0 = phases[0]
    new_phases = phases / phase0
    new_phase_angles = np.angle(new_phases)
    #print("\n[Step 5] New complex phases and related arguments:\n")
    #print("New complex phases: ", new_phases)
    #print("New phase angles [rad]:", new_phase_angles)
    #print("Global phase: ", phase0)

    # Step 6: Diagonal matrix with array "new_phases"
    diagonal_matrix = np.diag(new_phases)
    #print("\n[Step 6] Diagonal matrix from new complex phases:\n")
    #print(diagonal_matrix)

    return {
        "Magnitudes": magnitudes,
        "New complex phases": new_phases,
        "New phase angles": new_phase_angles,
        "Global phase": phase0,
        "Diagonal matrix": diagonal_matrix}

def qspParameters_mod(unit_vector : np.ndarray, n : int) -> tuple[list, list, list, list, list]:
    
    # PARAMETERS FOR LAMBDA2

    #assert np.isclose(np.linalg.norm(unit_vector), 1, atol = 1e-6), f"Coefficient vector norm is not 1: {np.linalg.norm(unit_vector)} != 1."

    #assert (len(unit_vector) == pow(2, n)), f"Wrong number of coefficients 'V' or wrong value of 'n': {len(unit_vector)} != {pow(2, n)}."

    #coefficient_matrix, angles, solution, variables, unused_variables = solve_system(unit_vector, n) #commented for opt_qsp_lambda2

    #a = solution[0::3] #commented for opt_qsp_lambda2
    #b = solution[1::3] #commented for opt_qsp_lambda2
    #d = solution[2::3] #commented for opt_qsp_lambda2

    #prefix_vector, suffix_vector = (a, b), d #commented for opt_qsp_lambda2

    theta_vector = generateThetaVect(unit_vector, n)  # [(2^n)-1] theta angles
    
    #print(f"Thetas generated from BST: {theta_vector}")
    #print(f"Prefix lambda angles (alphas, betas): {prefix_vector}")
    #print(f"Suffix lambda angles (deltas): {suffix_vector}")
    #print("_"*50)

    #params_vector = [prefix_vector, theta_vector, suffix_vector] #commented for opt_qsp_lambda2, replaced by the next line
    params_vector = theta_vector

    #prefix_alphas, middle_alphas, suffix_alphas = [], [], [] # Cannot use numpy because the resulting array has different shapes
    #previous line commented for opt_qsp_lambda2, replaced by the next line
    #middle_alphas = []
    
    #ucg_0_angles = [] #commented for opt_qsp_lambda2
    #global_phases = []

    for k in range(n): # Multiplexor decomposition
        #lambda_diagonals = []
        match k:
            case 0:
                #alpha0 = params_vector[0][0][0] #commented for opt_qsp_lambda2
                #global_phases.append(np.exp(1j*(alpha0))) #commented for opt_qsp_lambda2

                #beta0 = params_vector[0][1][0] #commented for opt_qsp_lambda2
                #print(f"Angle for Rz(beta0) of UCG {k+1}: {beta0}")
                #ucg_0_angles.append(beta0) #commented for opt_qsp_lambda2

                gamma0 = params_vector[0]
                #print(f"Angle for Ry(gamma0) of UCG {k+1}: {gamma0}")
                #ucg_0_angles.append(gamma0) #commented for opt_qsp_lambda2

                #delta0 = params_vector[2][0] #commented for opt_qsp_lambda2
                #print(f"Angle for Rz(delta0) of UCG {k+1}: {delta0}")
                #ucg_0_angles.append(delta0) #commented for opt_qsp_lambda2

                #print("_"*50)
            case _:

                #for i, pv in enumerate(params_vector): #commented for opt_qsp_lambda2

                    diag_rz = np.array([])

                    #match i:
                        #case 0:
                            #for p in range(pow(2, k)):
                                #a_angle = pv[0][pow(2, k) + p - 1]
                                #b_angle = pv[1][pow(2, k) + p - 1]
                                #diag = np.array([np.exp(1j*(a_angle - b_angle)), np.exp(1j*(a_angle + b_angle))])
                                #diag_rz = np.append(diag_rz, diag)
                        #case _:
                    for p in range(pow(2, k)):
                        angle = params_vector[pow(2, k) + p - 1]
                        diag = np.array([np.exp(-1j*(angle)), np.exp(1j*(angle))])
                        diag_rz = np.append(diag_rz, diag)
                    
                    #print(f"Diagonal matrix {i+1}/3 of UCG {k+1}: {diag_rz}")
                    #print(f"Angles for diagonal of matrix {i}/3 of UCG {k+1}: {[np.angle(x) for x in diag_rz]}")

                    #lamb = np.array([coeff / diag_rz[0] for coeff in diag_rz])
                    #lambda_diagonals.append(lamb)
                    #print(f"Exponential diagonal of Lambda {k+1}.{i+1}: {lamb}")

                    #global_phases.append(diag_rz[0])

                    #comb = [np.angle(x) for x in lamb]
                    #print(f"Angle combination for UCG {k+1}: {comb}")

                    #binary_matrix = generateInnProdMatrix(k+1)
                    #alphas = generateAlphaVect(binary_matrix, comb[1:])
                    
                    #print(f"Aphas for Lambda {k+1}.{i+1}: {alphas}")

                    #alpha_check = (pow(2, 1 - (k + 1)) * (2 * binary_matrix - np.ones((pow(2, (k + 1)) - 1,pow(2, (k + 1)) - 1))) @ comb[1:]) # (2^(1-n))*(2*A - J)*thetas

                    #assert np.allclose(alpha_check, alphas, atol = 1e-6), "Alpha values don't match the check: \n" + str(alphas) + " -> " + str(alpha_check)

                    #match i:
                        #case 0:
                            #prefix_alphas.append(alphas)
                        #case 1:
                    #middle_alphas.append(alphas)
                        #case 2:
                            #suffix_alphas.append(alphas)

                    #print("-"*50)
                #print("_"*50)

    # PARAMETERS FOR FINAL LAMBDA
    #risultati = matrixLambdaFinal(unit_vector)


    # Step 7: Solve the linear system for parallelized phases
    #phases_alphas = []

    #comb = risultati["New phase angles"]
    #print("\n[Step 7a] Checking the new phase angles that enter the linear system:\n")
    #print(comb)

    #binary_matrix = generateInnProdMatrix(n)
    #print("\n[Step 7b] Checking the coefficient matrix:\n")
    #print(binary_matrix)
    #phases_alphas = generateAlphaVect(binary_matrix, comb[1:])
    #alpha_check = (pow(2, 1 - n) * (2 * binary_matrix - np.ones((pow(2, n) - 1,pow(2, n) - 1))) @ comb[1:])
    #assert np.allclose(alpha_check, phases_alphas, atol = 1e-6), "Alpha values don't match the check: \n" + str(phases_alphas) + " -> " + str(alpha_check)
    #print("\n[Step 7c] Checking the alphas:\n")
    #print(phases_alphas)
    #print(type(phases_alphas))

    #last_global_phase = risultati["Global phase"]
    #global_phases.append(last_global_phase)
    #last_diagonal = risultati["New complex phases"]
    #lambda_diagonals.append(last_diagonal)

    #return prefix_alphas, middle_alphas, suffix_alphas, ucg_0_angles, global_phases, lambda_diagonals #commented for opt_qsp_lambda2
    return middle_alphas, phases_alphas, gamma0, global_phases, lambda_diagonals
"""


def solve_Mk_recursive(b : np.ndarray) -> np.ndarray:
    """
    k indicates the level of each UCG, starting from k=2 (first UCG on 2 qubits).
    b is the vector of known terms.
    x is the vector of unknowns, which are the parameters of each Z-block.
    Solve the system M_k @ x = b with M_k in {±1}^{2^k x (2^k-1)} defined by:
        M_k = [[M_{k-1}, +1,  M_{k-1}],
               [M_{k-1}, -1, -M_{k-1}]]
    without building M_k.

    Input:
      b: length 2^k (must be power of 2)
    Output:
      x: length 2^k-1 (solution)
    """
    b = np.asarray(b, dtype=float)
    m = b.size
    if m & (m - 1) != 0:
        raise ValueError("Length of b must be a power of 2.")
    if m == 2:
        # k=1: M_1 = [[+1],[-1]] so x = b[0] (and b[1] should be -b[0])
        return np.array([b[0]], dtype=float)

    half = m // 2
    p = b[:half]
    q = b[half:]

    s = 0.5 * (p + q)          # RHS for M_{k-1}@u, known term of the recursive system for u
    t = 0.5 * (p - q)          # RHS for c*1 + M_{k-1}@v, known term of the recursive system for v

    u = solve_Mk_recursive(s)  # length half-1

    c = float(np.mean(t))
    r = t - c                  # length half, sum(r)=0

    v = solve_Mk_recursive(r)  # length half-1

    return np.concatenate([u, np.array([c]), v])


def b_from_theta_level(theta_vector: np.ndarray, level_l: int) -> np.ndarray:
    """
    Build b for the modulus-UCG level 'l' (with l>=1).
    According to my convention in qspParameters_mod:

      angles = theta_have_index = theta_vector[2^l + p - 1],  p=0..2^l-1
      b = [-angle1,+angle1, -angle2,+angle2, ...]  (length 2^(l+1))

    This corresponds to a diagonal diag(exp(-i*angle), exp(+i*angle)) repeated.
    """
    if level_l < 1:
        raise ValueError("level_l must be >= 1 (level 0 is the single Ry).")

    theta_vector = np.asarray(theta_vector, dtype=float)
    angles = []
    for p in range(2**level_l):
        angles.append(theta_vector[(2**level_l) + p - 1])

    b = np.empty(2 * len(angles), dtype=float)
    for j, a in enumerate(angles):
        b[2*j] = -a
        b[2*j + 1] = +a
    return b


def srbb_params_moduli_from_theta(theta_vector: np.ndarray, n: int):
    """
    For each (modulus) ladder level l=1..n-1:
      - build b^{(l)} (length 2^(l+1))
      - solve M_{l+1} @ x = b^{(l)}  -> x has length 2^(l+1) - 1

    Returns list 'x_levels' where x_levels[l-1] is the SRBB parameter vector
    for UCG level l (acting on l+1 qubits).
    """
    x_levels = []
    for l in range(1, n):
        b = b_from_theta_level(theta_vector, l)
        x = solve_Mk_recursive(b)
        x_levels.append(x)
    return x_levels


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Map angles to (-pi, pi]."""
    return (x + np.pi) % (2*np.pi) - np.pi


def circular_mean_phase(phases: np.ndarray) -> float:
    """
    Circular mean of angles: arg(sum exp(i*phi)).
    Returns a value in (-pi, pi].
    """
    z = np.sum(np.exp(1j * phases))
    #print(z)
    #print(f"Circular mean: {z:+.9e}")
    # if z ~ 0, mean direction is undefined; choose 0
    if np.abs(z) < 1e-15:
        return 0.0
    return float(np.angle(z))


def b_from_state_phases_mean(unit_vector: np.ndarray, eps: float = 1e-14) -> tuple[np.ndarray, float, float]:
    """
    Build b_phase (len=2^N) for the linear system M_N @ x = b_phase,
    using a *clean* global phase choice based on the circular mean of phases.

    Steps:
      1) extract phases phi_j = arg(a_j); set phi_j=0 if |a_j|<eps (phase irrelevant)
      2) choose global phase g0 as circular mean of phi
      3) b = wrap(phi - g0)
      4) enforce solvability: sum(b)=0 by subtracting mean(b), then wrap again

    Returns:
      b_phase: float array length 2^N, with mean ~ 0 (exact up to floating error)
      g0: global phase removed (rad)
    """
    v = np.asarray(unit_vector, dtype=complex)
    r = np.abs(v)
    phi = np.angle(v)
    #print(phi)
    #print(f"Vector of complex phases: {phi}")

    # phase undefined if amplitude ~0 -> set to 0 (doesn't matter physically)
    phi = np.where(r < eps, 0.0, phi)

    g0 = circular_mean_phase(phi)
    #print(g0)
    #print(f"Circular mean phase g0: {g0:+.9e}")
    b0 = wrap_to_pi(phi - g0)
    #print(b)
    #print(f"Vector of known terms after -g0 and wrap, named b0: {b0}")

    # SU solvability constraint: 1^T b = 0
    mu  = float(np.mean(b0))
    #print(f"Mean on b0, named mu: {mu}")
    b1  = b0 - mu
    #b = wrap_to_pi(b)
    #print(b)
    #print(f"Vector of known terms after -mu, named b1: {b1}")
    alpha_total = g0 + mu              # fase globale complessiva rimossa
    #print(f"Total global phase (g0+mu) removed: {alpha_total}")

    #diag_equivalence_checks(phi, b1, alpha=alpha_total, label="explicit alpha")
    # e volendo anche la stima automatica (robusta a wrap / rumore):
    #diag_equivalence_checks(phi, b1, alpha=None, label="estimated alpha")

    return b1.astype(float), g0, alpha_total


def diag_equivalence_checks(phi, b_final, alpha=None, eps=1e-12, label=""):
    """
    phi: fasi originali (float), qualunque range
    b_final: fasi finali che passi al solver (float)
    alpha: fase globale totale rimossa (se la conosci). Se None, viene stimata.
    """

    D_target = np.exp(1j * phi)       # diag target (come vettore complesso)
    #print(f"\nD_target: {D_target}")
    D_out    = np.exp(1j * b_final)   # diag prodotta dal tuo schema (senza global phase)
    #print(f"D_out: {D_out}")

    # (A) Se non mi dai alpha, lo stimo nel modo più pulito:
    # voglio minimizzare || D_out - e^{-i alpha} D_target ||
    # alpha = arg( sum_k D_out_k * conj(D_target_k) )
    if alpha is None:
        z = np.vdot(D_target, D_out)  # = sum conj(D_target)*D_out
        alpha = -np.angle(z)          # perché voglio e^{-i alpha} D_target ≈ D_out

    #print(f"Global phase to build D_corr: {alpha} -> must correspond to go+mu")

    D_corr = np.exp(-1j * alpha) * D_target
    #print(f"D_corr: {D_corr}")

    err_inf = np.max(np.abs(D_out - D_corr))
    err_rms = np.sqrt(np.mean(np.abs(D_out - D_corr)**2))

    #print(f"\n[{label}] alpha used/estimated = {alpha:+.12e}")
    #print(f"[{label}] max|D_out - D_corr| = {err_inf:.3e}")
    #print(f"[{label}] rms|D_out - D_corr| = {err_rms:.3e}")

    # Extra: fase residua per componente (dovrebbe essere ~0 dopo correzione globale)
    residual = np.angle(D_out * np.conj(D_corr))  # arg( D_out / D_corr )
    residual = wrap_to_pi(residual)
    #print(f"\n[{label}] residual phase stats (rad): "f"max={np.max(np.abs(residual)):.3e}, mean={np.mean(residual):+.3e}")

    ok = err_inf < eps
    #print(f"[{label}] equivalent up to global phase? {ok}")

    score, ov = check_overlap(D_target, D_out)
    #print("\nGlobal-phase-invariant score =", score)
    #print("Estimated alpha =", -np.angle(ov))
    
    return alpha, err_inf, residual

def check_overlap(D_target, D_out):
    # entrambi sono vettori complessi con |component|=1
    ov = np.vdot(D_target, D_out)          # sum conj(D_target)*D_out
    score = np.abs(ov) / len(D_target)     # in [0,1]
    return score, ov





### TEST FUNCTIONS for linear systems ###

def build_Mk_explicit(k: int) -> np.ndarray:
    """
    Build M_k in {±1}^{2^k x (2^k-1)} explicitly using:
      M_k = [[M_{k-1}, +1,  M_{k-1}],
             [M_{k-1}, -1, -M_{k-1}]]
    Base: M_1 = [[+1],[-1]]
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return np.array([[+1.0], [-1.0]])

    M_prev = build_Mk_explicit(k - 1)  # shape (2^(k-1), 2^(k-1)-1)
    rows = M_prev.shape[0]
    ones_col = np.ones((rows, 1), dtype=float)

    top = np.hstack([M_prev, +ones_col, +M_prev])
    bot = np.hstack([M_prev, -ones_col, -M_prev])
    return np.vstack([top, bot])


def check_level_solution_explicit(b: np.ndarray, x: np.ndarray, *, atol=1e-9, rtol=1e-9):
    """
    Checks M_k @ x = b for the inferred k from len(b)=2^k.
    Returns (ok, max_abs_err, residual_vector).
    """
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)

    m = b.size
    if m & (m - 1) != 0:
        raise ValueError("len(b) must be power of 2")
    k = int(np.log2(m))

    M = build_Mk_explicit(k)
    bx = M @ x
    resid = bx - b
    max_abs = float(np.max(np.abs(resid)))
    ok = np.allclose(bx, b, atol=atol, rtol=rtol)
    return ok, max_abs, resid


def Mk_row(k: int, i: int) -> np.ndarray:
    """
    Returns the i-th row (0-based) of M_k as a ±1 vector of length 2^k - 1,
    without building the whole matrix.
    """
    if k == 1:
        return np.array([+1.0]) if i == 0 else np.array([-1.0])

    half = 2 ** (k - 1)
    if i < half:
        r = Mk_row(k - 1, i)
        return np.concatenate([r, np.array([+1.0]), r])
    else:
        r = Mk_row(k - 1, i - half)
        return np.concatenate([r, np.array([-1.0]), -r])


def check_level_solution_subset(b: np.ndarray, x: np.ndarray, *, num_rows=20, seed=0, atol=1e-9):
    """
    Checks a random subset of rows of M_k @ x = b.
    """
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)

    m = b.size
    if m & (m - 1) != 0:
        raise ValueError("len(b) must be power of 2")
    k = int(np.log2(m))

    rng = np.random.default_rng(seed)
    idx = rng.choice(m, size=min(num_rows, m), replace=False)

    max_abs = 0.0
    ok = True
    for i in idx:
        row = Mk_row(k, int(i))
        val = float(row @ x)
        err = abs(val - b[i])
        max_abs = max(max_abs, err)
        if err > atol:
            ok = False
    return ok, max_abs, idx
