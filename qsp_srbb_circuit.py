import pennylane as qml
#from pennylane import numpy as pnp
import numpy as np
#import matplotlib.pyplot as plt
from pennylane.templates.embeddings import AmplitudeEmbedding
#import config
#from pennylane_qiskit.remote import RemoteDevice
import srbb_circuit
from pennylane.workflow import construct_tape

#Exact version of QSP algorithm with SRBB (only diagonal contributions)

"""
def make_circuit_qnode(dev, n_qubit):
	@qml.qnode(dev)
	def circuit(params, x_max, rot_count, x=[], U_approx=[], initial_state = False, state = False):
	
		#Create the circuit from gray code

		#Args:
			#params (np.array): theta angles
			#x_max (int): number of ProdT_factors and M_factors
			#rot_count (int): num of rots
			#x (): state to be encoded. if trace or fidelity are used as loss function
			#U_approx (np.array): approximated U
			#state (bool): if false train the modulo, if true train the phase 
	

		

		if initial_state: #srbb
			binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection = srbb_circuit.srbb_initialization(x_max, n_qubit)
		else:
			simpli_Gray_collection=Simplification_Gray_matrix(Gray_matrix(n_qubit-1,False), n_qubit)
		#if initial_state is not None:
		#	AmplitudeEmbedding(initial_state, range(n_qubit), normalize=True)

		#if U_approxIS is not None:
		#	qml.QubitUnitary(U_approxIS, wires = range(n_qubit))
		
		#if there is no U, then i want to train the network to achieve it
		if len(U_approx) == 0:
			#if fidelity or trace distance are used as loss function, encodes the state
			if len(x) != 0:
				AmplitudeEmbedding(x, range(n_qubit), normalize = True)
				#qml.Barrier(wires = [0, 1])
			
			#add the VQC
			if initial_state: #srbb
				srbb_circuit.circuit(params, binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection, x_max, rot_count, n_qubit)
			else:
				SU_approx_circuit(params, x_max, simpli_Gray_collection, state, n_qubit)
			
			if len(x) == 0:
				return qml.density_matrix(range(n_qubit))
			else:
				return qml.density_matrix(range(n_qubit)), qml.state(), qml.probs(range(n_qubit))
			
		else: #testing
			if len(x) != 0:
				AmplitudeEmbedding(x, range(n_qubit), normalize = True)

			#based on if the function has the parameters or the matrix, generate the circuit
			if len(params) == 0:
				qml.QubitUnitary(U_approx, wires = range(n_qubit))
			else:
				SU_approx_circuit(params, x_max, simpli_Gray_collection, state, n_qubit)			

			if isinstance(dev, RemoteDevice):
				return qml.counts(wires = range(n_qubit)), qml.counts(wires =range(n_qubit)), qml.probs(range(n_qubit))
			else:
				return qml.state(), qml.probs(range(n_qubit)), qml.density_matrix(range(n_qubit))
	return circuit
"""	


def amplitude_density(dev, n_qubit):
	@qml.qnode(dev)
	def circuit(x):
		"""
		Retrieve Density matrix of state

		Args:
			x (): state to be encoded

		Returns:
			(): density matrix
		"""
		AmplitudeEmbedding(x, range(n_qubit), normalize = True)
		return qml.density_matrix(range(n_qubit))
	return circuit


#Format useful for checks
#def format_matrix(matrix):
#	M=('\n'.join(''.join('{:11}'.format(np.round(item,1)) for item in row) for row in matrix))
#	return M


def Gray_matrix(n_bits,side=bool):
	"""
		Used for ZETA_factor
		
		Args:
			n_bits (int): number of bits
			side (bool): True for right decomposition, False for left decomposition

		Returns:
			(): gray matrix collection
	"""
	Gray_matrix_collection=[]
	if side==True:
		Gray_matrix_1=np.array([[0],[1]])
		Gray_matrix_collection.append(Gray_matrix_1)
		Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
		for i in range(2,n_bits+1):
			Gray_matrix_next=np.empty(shape=(2**i,i),dtype=int)
			semi_rows=2**(i-1)
			Gray_matrix_next[0:semi_rows,0]=[0]*semi_rows
			Gray_matrix_next[0:semi_rows,1:i]=Gray_matrix_1
			Gray_matrix_next[semi_rows:2**i,0]=[1]*semi_rows
			Gray_matrix_next[semi_rows:2**i,1:i]=Gray_matrix_1reverse
			Gray_matrix_collection.append(Gray_matrix_next)
			Gray_matrix_1=Gray_matrix_next
			Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
	else:
		Gray_matrix_1=np.array([[1],[0]])
		Gray_matrix_collection.append(Gray_matrix_1)
		Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
		for i in range(2,n_bits+1):
			Gray_matrix_next=np.empty(shape=(2**i,i),dtype=int)
			semi_rows=2**(i-1)
			Gray_matrix_next[0:semi_rows,0]=[1]*semi_rows
			Gray_matrix_next[0:semi_rows,1:i]=Gray_matrix_1
			Gray_matrix_next[semi_rows:2**i,0]=[0]*semi_rows
			Gray_matrix_next[semi_rows:2**i,1:i]=Gray_matrix_1reverse
			Gray_matrix_collection.append(Gray_matrix_next)
			Gray_matrix_1=Gray_matrix_next
			Gray_matrix_1reverse=np.flip(Gray_matrix_1,0)
	return Gray_matrix_collection


def Simplification_Gray_matrix(Gray_matrix_collection, n_qubit):
	"""
		Inside ZETA_factor, Simplify the CNOT number
		CNOT parameters depend on this matrix

		Args:
			Gray_matrix_collection ():

		Returns:
			(): simplified gray
	"""
	simpli_Gray_collection=[]
	for p in range(n_qubit-1):
		Gray_matrix_p=Gray_matrix_collection[p]
		extension=Gray_matrix_p[0]
		Gray_matrix_extended=np.empty(shape=(2**(p+1)+1,p+1))
		Gray_matrix_extended[0:2**(p+1),0:p+1]=Gray_matrix_p
		Gray_matrix_extended[2**(p+1),0:p+1]=extension
		simpli_Gray_matrix=np.empty(shape=(2**(p+1),p+1),dtype=int)
		for i in range(2**(p+1)):
			m1=Gray_matrix_extended[i]
			m2=Gray_matrix_extended[i+1]
			for j in range(p+1):
				if m1[j]==m2[j]:
					simpli_Gray_matrix[i][j]=0
				else:
					simpli_Gray_matrix[i][j]=1
		simpli_Gray_collection.append(simpli_Gray_matrix)
	
	return simpli_Gray_collection

"""
def Theta_array_gen(state, n_qubit):

	#Generates the total number of parameters for the circuit

	#Args:
		#state (bool): True if train phase, False if train modulo

	#Returns:
		#params (np.array): trainable parameters
		#rot_count_ZETA (int): number of rot 


	#ZETA FACTOR -> exp{3,8,15,24...}
	if not state:
		rot_count_ZETA = 1
		for n in range(2, n_qubit+1):
			rot_count_ZETA += 2**n - 1
	else:
		rot_count_ZETA = 2**n_qubit -1

	params=np.random.randn(rot_count_ZETA, requires_grad=True)
	return params,rot_count_ZETA
"""
	
def ZETA_factor2qubits(theta_collection_ZETA, delete = bool, modulo = True):
	"""
	ZETA_factor definition for 2 qubits (special case outside the scalability scheme)

	Args:
		theta_collection_ZETA (): theta parameters
		delete (bool)
		modulo (bool): if modulo add two final gates
	"""
	if modulo:
		qml.QubitUnitary([[1, 0], [0, -1j]], wires = 1)
		qml.Hadamard(1)
	if delete==False:
		qml.CNOT(wires=[0,1])
		qml.RZ(2*theta_collection_ZETA[0],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(2*theta_collection_ZETA[1],wires=[0])
		qml.RZ(2*theta_collection_ZETA[2],wires=[1])
	else:
		qml.RZ(2*theta_collection_ZETA[0],wires=[1])
		qml.CNOT(wires=[0,1])
		qml.RZ(2*theta_collection_ZETA[1],wires=[0])
		qml.RZ(2*theta_collection_ZETA[2],wires=[1])
	if modulo:
		qml.Hadamard(1)
		qml.S(1)
	


def ZETA_factor(theta_collection_ZETA,simpli_Gray_collection, n, modulo = True):
	"""
	ZETA_factor definition for n>2 qubits
	
	Args:
		theta_collection_ZETA (): theta parameters
		simpli_gray_collection (): simplified gray
		n (int): qubits
		modulo (bool): true for modulo, false for phase
	"""
	offset=0
	if modulo:
		qml.QubitUnitary([[1, 0], [0, -1j]], wires = n-1)
		qml.Hadamard(n-1)
	for k in range(n - 1):
		for i in range(simpli_Gray_collection[n-2-k].shape[0]):
			for j in range(simpli_Gray_collection[n-2-k].shape[1]):
				if simpli_Gray_collection[n-2-k][i][j]==1:
					qml.CNOT(wires=[j,n-1-k])
					qml.RZ(2*theta_collection_ZETA[offset + i],wires=[n-1-k])		
		offset += simpli_Gray_collection[n-2-k].shape[0]		
	qml.RZ(2*theta_collection_ZETA[-1],wires=[0])
	if modulo:
		qml.Hadamard(n-1)
		qml.S(n-1)
	

def SU_approx_circuit(params, x_max, simpli_Gray_collection, state, n_qubit):
	"""
		Generate the VQC

		Args:
			params (np.array): theta angles
			x_max (int): number of ProdT_factors and M_factors
			simpli_gray_collection (): simplified gray
			state (bool): true for phase circuit, false for modulo

	"""
	
	if not state:
		theta_collection_ZETA = params[1:]
		qml.RY(params[0], wires = 0)
	else:
		theta_collection_ZETA = params


	if n_qubit == 2:
		
		#VQC modulo
		if not state:
			theta_zeta_2 = theta_collection_ZETA[0:3]
			theta_zeta_2 = reorder_x_for_zblock(theta_zeta_2, 2)
			ZETA_factor2qubits(theta_zeta_2, False) # modulo=True default
		else:
			#VQC phase
			theta_zeta_2 = reorder_x_for_zblock(theta_collection_ZETA, 2)
			ZETA_factor2qubits(theta_zeta_2, False, False) # modulo=False for phase
	else:
		#ZETA FACTOR
		if not state:
			previous_index = 0
			for n in range(2, n_qubit+1):
				end_index = previous_index + 2**n - 1
				#print(n)
				theta_zeta_n = theta_collection_ZETA[previous_index: end_index]
				theta_zeta_n = reorder_x_for_zblock(theta_zeta_n, n)
				#print(theta_zeta_n)
				#input()
				if n == 2:
					ZETA_factor2qubits(theta_zeta_n, False)
				else:
					ZETA_factor(theta_zeta_n,simpli_Gray_collection[:n-1], n)
				previous_index = end_index
		else:
			#VQC phase
			theta_zeta_max = reorder_x_for_zblock(theta_collection_ZETA, n_qubit)
			ZETA_factor(theta_zeta_max,simpli_Gray_collection, n_qubit, False)










def gray_code_bits(n: int):
    if n <= 0:
        return [""]
    codes = ["0", "1"]
    for _ in range(2, n + 1):
        codes = ["0" + c for c in codes] + ["1" + c for c in reversed(codes)]
    return codes

def zblock_param_order_indices(N: int):
	# base: N=2 -> [3,2,1] (θ15, θ8, θ3)
	if N < 2:
		return []
	if N == 2:
		return [3, 2, 1]
	
	G = gray_code_bits(N)
	odd = [int(bits, 2) for bits in G if bits[-1] == "1"]  # LSB=1

	# regola tua: "parto dalla seconda riga, ciclico"
	if len(odd) > 1:
		odd = odd[1:] + odd[:1]
	
	prev = zblock_param_order_indices(N - 1)
	prev_appended0 = [2 * i for i in prev]
	
	order = odd + prev_appended0
	i_star = 2**(N-1)  # bitstring 100..0
	if i_star in order:
		order.remove(i_star)
		order.append(i_star)
	
	return order

def expected_rz_labels_for_zblock(N: int):
    """
    Lista delle label θ_{m^2-1} nell'ordine con cui dovrebbero essere consumate
    dal blocco Z_N (quello riordinato da reorder_x_for_zblock).
    """
    order_i = zblock_param_order_indices(N)  # i in 1..2^N-1
    return [theta_label_from_i(i) for i in order_i]

def reorder_x_for_zblock(x, N: int):
    """
    x: array len = 2^N - 1, in ordine crescente per indice i=1..2^N-1
    return: array stesso len, ma nell'ordine in cui ZETA_factor/ZETA_factor2qubits lo consumano
    """
    order_i = zblock_param_order_indices(N)
    return [x[i-1] for i in order_i]


def print_su_approx_circuit(params, x_max, simpli_Gray_collection, state, n_qubit, dev_name="default.qubit"):
    """
    Debug utility: stampa il circuito finale costruito da SU_approx_circuit.
	state=True per phase -> params=x_phase
	state=False per modulo -> params=paramsMod
    """
    dev = qml.device(dev_name, wires=n_qubit)

    @qml.qnode(dev)
    def circuit():
        SU_approx_circuit(params, x_max, simpli_Gray_collection, state, n_qubit)
        return qml.state()

    drawer = qml.draw(circuit)
    print(drawer())

def theta_label_from_i(i: int) -> int:
    # θ_{m^2-1}, con m=i+1
    m = i + 1
    return m*m - 1

def make_x_label_params(N: int) -> np.ndarray:
    """
    Ritorna x_label (len=2^N-1) IN ORDINE NATURALE (i crescente),
    ma tale che, quando applichi RZ(2*x_label), sul disegno compare la label intera.
    """
    x = np.zeros(2**N - 1, dtype=float)
    for i in range(1, 2**N):
        x[i-1] = theta_label_from_i(i) / 2.0
    return x

def extract_rz_sequence(params, simpli_Gray_collection, state, n_qubit, dev_name="default.qubit"):
    """
    Costruisce il circuito SU_approx_circuit e ritorna la sequenza (in ordine)
    degli angoli usati nei gate RZ presenti nel tape.
    """
    dev = qml.device(dev_name, wires=n_qubit)

    @qml.qnode(dev)
    def circuit():
        SU_approx_circuit(params, x_max=None, simpli_Gray_collection=simpli_Gray_collection,
                          state=state, n_qubit=n_qubit)
        return qml.state()

    # costruisci il tape senza dover interpretare output
    tape = construct_tape(circuit)()

    rz_angles = []
    for op in tape.operations:
        # PennyLane usa name="RZ" per qml.RZ
        if op.name == "RZ":
            rz_angles.append(float(op.parameters[0]))
    return rz_angles






"""
def test(dev, n_qubit):
	@qml.qnode(dev)

	def circuit(U, paramsMod, paramsPhase, state, dm, rot_countMod, rot_countPhase, x_max, hw):
		
		#Test the optimized VQC

		#Args:
		#	U (np.array): approximated U
		#	paramsMod (np.array): optimized modulo theta
		#	paramsPhase (np.array): optimized phase theta
		#	state (bool): true for precise state with phase, false for probabilities
		#	dm (): density matrix of state
		#	rot_countMod (int): number of rot for modulo
		#	rot_countPhase (int): number of rot for phase
		#	x_max (int): number of ProdT_factors and M_factors
		#	hw (str): true for hardware execution

		

		if len(U) != 0:
			qml.QubitUnitary(U, wires = range(n_qubit))
		else:
			simpli_Gray_collection=Simplification_Gray_matrix(Gray_matrix(n_qubit-1,False))
			SU_approx_circuit(paramsMod, x_max, simpli_Gray_collection, False)
			if state:
				SU_approx_circuit(paramsPhase, x_max, simpli_Gray_collection, True)			

		if state:
			if hw:
				return qml.probs(wires = range(n_qubit))
			if not dm:
				return qml.state()
			else: 
				return qml.density_matrix(wires = range(n_qubit))
		else:
			return qml.probs(wires = range(n_qubit))
	return circuit
"""

"""
def test_srbb_method(dev, n_qubit):
	@qml.qnode(dev)
	def circuit(U, state, U_approxIS = None, initial_state = None, dm = False):
		
		#Test VQC

		#Args:
		#	U (np.array): approximated U
		#	state (bool): true for precise state, false for probabilities
		#	dm (bool): True to retrieve density matrix for trace distance computation
		

		if initial_state is not None:
			AmplitudeEmbedding(initial_state, range(n_qubit), normalize=True)
		
		if U_approxIS is not None:
			qml.QubitUnitary(U_approxIS, wires = range(n_qubit))


		qml.QubitUnitary(U, wires = range(n_qubit))

		if state:
			if not dm:
				return qml.state()
			else: 
				return qml.density_matrix(wires = range(n_qubit))
		else:
			return qml.probs(wires = range(n_qubit))
	return circuit
"""


def test_srbb_method_params(dev, n_qubit):
	@qml.qnode(dev)
	def circuit(paramsMod, paramsPhase, state, paramsModIS = None, paramsPhaseIS = None, rot_countIS = None, initial_state = None, dm = False):
		"""
		Test VQC

		Args:
			U (np.array): approximated U
			state (bool): true for precise state, false for probabilities
			dm (bool): True to retrieve density matrix for trace distance computation
		"""

		x_max=2**(n_qubit-1)-1

		if initial_state is not None:
			#print("AA")
			AmplitudeEmbedding(initial_state, range(n_qubit), normalize=True)
		
		if paramsModIS is not None:
			simpli_Gray_collection=Simplification_Gray_matrix(Gray_matrix(n_qubit-1,False), n_qubit)
			SU_approx_circuit(paramsPhaseIS, x_max, simpli_Gray_collection, True, n_qubit)

			binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection = srbb_circuit.srbb_initialization(x_max, n_qubit)
			srbb_circuit.circuit(paramsModIS, binary_matrix_reverse, k_array, simpli_matrix_even, simpli_matrix_odd, simpli_Gray_collection, x_max, rot_countIS, n_qubit)
			
			

		

		simpli_Gray_collection=Simplification_Gray_matrix(Gray_matrix(n_qubit-1,False), n_qubit)
		SU_approx_circuit(paramsMod, x_max, simpli_Gray_collection, False, n_qubit)

		SU_approx_circuit(paramsPhase, x_max, simpli_Gray_collection, True, n_qubit)


		if state:
			if not dm:
				return qml.state()
			else: 
				return qml.density_matrix(wires = range(n_qubit))
		else:
			return qml.probs(wires = range(n_qubit))
	return circuit


def test_param_order_on_circuit(N: int, state: bool):
    """
    state=True  -> phase max
    state=False -> ladder modulo (include RY + blocchi Z per n=2..N)
    """
    simpli = Simplification_Gray_matrix(Gray_matrix(N-1, False), N)

    # parametri etichetta: se state=True serve len=2^N-1
    # se state=False serve [RY0] + concatenazione dei blocchi n=2..N
    if state:
        params = make_x_label_params(N)  # len 2^N-1
    else:
        # costruiamo un params finto coerente col tuo formato:
        # params[0] = RY iniziale (mettiamo 0), params[1:] = theta_collection_ZETA
        # theta_collection_ZETA deve contenere concat dei blocchi n=2..N, ciascuno lungo (2^n-1) in ordine naturale.
        blocks = []
        for n in range(2, N+1):
            blocks.append(make_x_label_params(n))  # len 2^n-1
        theta_concat = np.concatenate(blocks) if blocks else np.array([], dtype=float)
        params = np.concatenate(([0.0], theta_concat))

    rz_measured = extract_rz_sequence(params=params, simpli_Gray_collection=simpli,
                                     state=state, n_qubit=N)

    print("\n" + "="*70)
    print(f"TEST ordine parametri sul circuito | N={N} | state={'PHASE' if state else 'MODULO'}")
    print(f"Numero RZ nel tape: {len(rz_measured)}")
    print("-"*70)

    if state:
        expected = expected_rz_labels_for_zblock(N)  # tutte le RZ del blocco max
        # In modalità phase il blocco Z_N dovrebbe usare esattamente (2^N-1) RZ
        print(f"Atteso (count): {len(expected)}  | Misurato (count): {len(rz_measured)}")
        ok_len = (len(expected) == len(rz_measured))
        ok_seq = ok_len and all(int(round(a)) == e for a, e in zip(rz_measured, expected))

        print(f"LEN ok: {ok_len} | SEQ ok: {ok_seq}")
        if not ok_seq:
            # stampa compatta dei primi mismatch
            for k, (a, e) in enumerate(zip(rz_measured, expected)):
                if int(round(a)) != e:
                    print(f"Mismatch k={k}: misurato RZ({a}) vs atteso RZ({e})")
                    break
        else:
            print("Sequenza RZ corrisponde all'ordine atteso (label).")

        # stampa prime righe per ispezione manuale
        show = min(20, len(expected))
        print("\nPrime righe (k | misurato | atteso):")
        for k in range(show):
            print(f"{k:>3} | {int(round(rz_measured[k])):>4} | {expected[k]:>4}")

    else:
        # In modalità modulo ci sono più blocchi (n=2..N): stampiamo per blocco.
        idx = 0
        for n in range(2, N+1):
            expected_n = expected_rz_labels_for_zblock(n)
            m = len(expected_n)
            measured_n = [int(round(x)) for x in rz_measured[idx: idx+m]]
            ok = (measured_n == expected_n)

            print(f"\nBlocchetto n={n}: attesi {m} RZ | ok={ok}")
            if not ok:
                # mostra mismatch
                for k, (a, e) in enumerate(zip(measured_n, expected_n)):
                    if a != e:
                        print(f"  Mismatch k={k}: misurato {a} vs atteso {e}")
                        break
            else:
                # mostra prime 8 per conferma
                show = min(8, m)
                print("  prime:", list(zip(measured_n[:show], expected_n[:show])))

            idx += m

        print(f"\nTotale RZ consumati nei blocchi n=2-{n}, N: {idx}")
        print("Nota: qui stiamo confrontando solo la parte 'diagonale Z' (RZ).")



if __name__ == "__main__":

    # =========================
    # DEBUG CONTROL PANEL
    # =========================
    N = 3
    mode = "modulo"             # "phase" or "modulo"
    use_labels = True          # True -> usa make_x_label_params (RZ mostra 15,8,3,...)
    do_seq_test = True         # True -> chiama test_param_order_on_circuit
    draw_circuit = True        # True -> stampa il circuito

    # =========================
    # Helpers per params
    # =========================
    def build_phase_params(N, use_labels=True):
        if use_labels:
            return make_x_label_params(N)  # len = 2^N - 1
        # dummy numerico
        return np.arange(1, 2**N) * 0.1

    def build_modulo_params(N, use_labels=True, ry0=0.0):
        # MODULO ladder: params = [RY0] + concat_{n=2..N} (blocchi Z_n)
        blocks = []
        for n in range(2, N + 1):
            if use_labels:
                blocks.append(make_x_label_params(n))          # len = 2^n - 1
            else:
                blocks.append(np.arange(1, 2**n) * 0.1)        # dummy per blocco
        theta_concat = np.concatenate(blocks) if blocks else np.array([], dtype=float)
        return np.concatenate(([ry0], theta_concat))

    # =========================
    # Build Gray simplification
    # =========================
    # Nota: per N=2 non serve davvero, ma va bene lo stesso
    simpli_Gray_collection = Simplification_Gray_matrix(Gray_matrix(N - 1, False), N)

    # =========================
    # Run selected debug
    # =========================
    print("\n" + "#" * 80)
    print(f"### DEBUG | N={N} | mode={mode.upper()} | labels={use_labels} ###")

    if do_seq_test:
        if mode == "phase":
            test_param_order_on_circuit(N, state=True)
        elif mode == "modulo":
            test_param_order_on_circuit(N, state=False)
        else:
            raise ValueError("mode must be 'phase' or 'modulo'")

    if draw_circuit:
        if mode == "phase":
            params = build_phase_params(N, use_labels=use_labels)
            print("\n=== DRAW: SU_approx_circuit (PHASE) ===")
            print_su_approx_circuit(
                params=params,
                x_max=None,
                simpli_Gray_collection=simpli_Gray_collection,
                state=True,
                n_qubit=N,
            )

        elif mode == "modulo":
            params = build_modulo_params(N, use_labels=use_labels, ry0=0.0)
            print("\n=== DRAW: SU_approx_circuit (MODULO) ===")
            print_su_approx_circuit(
                params=params,
                x_max=None,
                simpli_Gray_collection=simpli_Gray_collection,
                state=False,
                n_qubit=N,
            )

# Note:
# - Ordering and assignment of Z-block parameters validated for N=2,3,4
# - use_labels=True is the reference debug mode
# - wire 0 convention enforced via i_star = 2**(N-1)
