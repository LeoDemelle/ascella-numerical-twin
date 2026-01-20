"""
Code réalisé par Léo Demelle, Jules Sayad-Barth et Jiaming Miao

Minimal Digital Twin of the Ascella photonic platform (MVP)

Architecture:
- Fock-state input preparation
- Universal linear interferometer (N_MODES x N_MODES unitary)
- Exact Fock-state simulation (Strawberry Fields backend)
- Classical Monte Carlo sampling
- Classical post-selection

Dependencies:
- strawberryfields
- numpy
"""

import time
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Fock, Interferometer, Ket
import streamlit as st

# ============================================================
# GLOBAL CONFIGURATION (CHANGE HERE FOR FAST TESTS)
# ============================================================

N_MODES = 4      # ⚠️ change to 6 or 4 for fast debugging
DEFAULT_CUTOFF = 4


# ============================================================
# LOGGING UTIL
# ============================================================

def log(msg, level=0):
    indent = "│   " * level
    print(f"{indent}├─ {msg}")


# ============================================================
# 1. INPUT STATE PREPARATION
# ============================================================

def tensor_ket(states):
    out = states[0]
    for s in states[1:]:
        out = np.kron(out, s)
    return out

def prepare_input_state(program, input_state):
    """
    Prepare a Fock input state.

    Args:
        program (sf.Program)
        input_state (list[int]): length = N_MODES
    """
    if len(input_state) != cutoff ** N_MODES:
        raise ValueError(
            f"Input state must have length {cutoff ** N_MODES}, "
            f"got {len(input_state)}"
        )

    with program.context as q:
        Ket(input_state) | q


# ============================================================
# 2. INTERFEROMETER LAYER
# ============================================================

def apply_interferometer(program, unitary):
    """
    Apply a universal linear interferometer.

    Args:
        program (sf.Program)
        unitary (np.ndarray): shape (N_MODES, N_MODES)
    """
    if unitary.shape != (N_MODES, N_MODES):
        raise ValueError(
            f"Unitary must be of shape ({N_MODES}, {N_MODES}), "
            f"got {unitary.shape}"
        )

    with program.context as q:
        Interferometer(unitary) | q


# ============================================================
# 3. ENGINE EXECUTION (EXACT + CLASSICAL SAMPLING)
# ============================================================

def run_experiment(program, cutoff_dim=DEFAULT_CUTOFF, shots=100):
    """
    Run the experiment using the Fock backend and sample classically.

    Returns:
        list[list[int]]: Fock samples
    """
    log("Initializing Fock backend", level=2)
    engine = sf.Engine(
        backend="fock",
        backend_options={"cutoff_dim": cutoff_dim}
    )

    log("Running quantum program (exact state evolution)", level=2)
    t0 = time.time()
    result = engine.run(program)
    t1 = time.time()
    log(f"Quantum evolution finished in {t1 - t0:.2f} s", level=3)

    state = result.state

    hilbert_dim = cutoff_dim ** N_MODES
    log("Computing full Fock probability distribution", level=2)
    log(f"Hilbert space dimension = {hilbert_dim}", level=3)

    t0 = time.time()
    probs = state.all_fock_probs().flatten()
    t1 = time.time()
    log(f"Probabilities computed in {t1 - t0:.2f} s", level=3)

    probs /= np.sum(probs)

    log("Generating Fock basis states", level=2)
    basis_states = list(
        np.ndindex(*([cutoff_dim] * N_MODES))
    )

    log(f"Sampling {shots} shots (classical Monte Carlo)", level=2)
    indices = np.random.choice(
        len(basis_states),
        size=shots,
        p=probs
    )

    samples = [list(basis_states[i]) for i in indices]
    log("Sampling complete", level=2)

    return samples


# ============================================================
# 4. POST-SELECTION
# ============================================================

def postselect_samples(samples, target_photon_number):
    """
    Post-select samples with a fixed total photon number.
    """
    return [s for s in samples if sum(s) == target_photon_number]


# ============================================================
# 5. HIGH-LEVEL EXPERIMENT WRAPPER
# ============================================================

def ascella_mvp_experiment(
    input_state,
    unitary,
    cutoff_dim=DEFAULT_CUTOFF,
    shots=100,
    postselect_n_photons=None
):
    """
    Complete MVP pipeline:
    Input -> Interferometer -> Exact simulation -> Sampling -> Post-selection
    """
    log("Initializing Ascella MVP experiment", level=0)

    prog = sf.Program(N_MODES)

    log("Preparing input Fock state", level=1)
    prepare_input_state(prog, input_state)

    log("Applying interferometer", level=1)
    apply_interferometer(prog, unitary)

    log("Executing quantum simulation", level=1)
    samples = run_experiment(
        prog,
        cutoff_dim=cutoff_dim,
        shots=shots
    )

    if postselect_n_photons is not None:
        log(
            f"Post-selecting samples with total photon number = "
            f"{postselect_n_photons}",
            level=1
        )
        before = len(samples)
        samples = postselect_samples(samples, postselect_n_photons)
        after = len(samples)
        log(f"Kept {after}/{before} samples", level=2)

    log("Experiment finished", level=0)
    return samples


# ============================================================
# 6. UTILITY: RANDOM HAAR UNITARY
# ============================================================

def random_haar_unitary(n=N_MODES, seed=None):
    """
    Generate a Haar-random unitary matrix of size n x n.
    """
    if seed is not None:
        np.random.seed(seed)

    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph


# ============================================================
# 7. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    st.title("Ascella Numerical Twin")

    log("Starting test run", level=0)
    cutoff = DEFAULT_CUTOFF
    N_MODES = st.slider("Choose the number of modes", 2, 12, 4)

    mode = st.selectbox("Choose the mode of computing", ["test","operation","vqe"], index=None)

    if mode == "test":
        #states = np.zeros((N_MODES, cutoff))
        states = np.array([
            [0,1,0,0],
            [0,1,0,0],
            [0,1,0,0],
            [0,1,0,0]
        ])

        log("Generating Haar-random interferometer", level=1)
        U = random_haar_unitary(seed=42)
    
    elif mode == "operation":
        st.subheader("Input state (Fock occupation per mode)")

        states = np.zeros((N_MODES, cutoff))
        for i in range(N_MODES):
            cols = st.columns(cutoff)
            for j in range(cutoff):
                states[i, j] = cols[j].number_input(
                    f"Mode {i}: |{j}>",
                    value=float(states[i, j].real),
                    step=0.1,
                    key=f"states_{i}_{j}"
                )
        
                
        st.subheader("Unitary transformation U")
        U = np.identity(N_MODES, dtype=complex)

        for i in range(N_MODES):
            cols = st.columns(N_MODES)
            for j in range(N_MODES):
                U[i, j] = cols[j].number_input(
                    f"U[{i},{j}]",
                    value=float(U[i, j].real),
                    step=0.1,
                    key=f"U_{i}_{j}"
                )

    else:
        st.write("Mode not chosen yet")

    
    
    st.write("Input Fock state:")
    st.write(states)
    input_state = tensor_ket(states)
    st.write("Unitary matrix U:")
    st.write(U)
    log(f"Input state: {input_state}", level=1)
    log(f"Unitary matrix: {U}", level=1)


    # EXPERIENT
    if st.button("Run experiment"):
        log("Launching Ascella MVP experiment", level=1)
        samples = ascella_mvp_experiment(
            input_state=input_state,
            unitary=U,
            cutoff_dim=DEFAULT_CUTOFF,
            shots=50,                 # 🔥 reduce for fast tests
            postselect_n_photons=None #sum(input_state)
        )

        log(f"Number of post-selected samples: {len(samples)}", level=1)

        if samples:
            log("First 5 samples:", level=1)
            for s in samples[:5]:
                print("│   │   ", s)
