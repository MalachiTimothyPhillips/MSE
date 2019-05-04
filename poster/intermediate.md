Algorithm:

  1. Divide $\Omega$ into $E$ nonoverlapping subdomains, such that $\Omega=\bigcap_{e=1}^{E} \Omega ^e$.
  2. Assign non-unique mortars to the surfaces of the subdomains.
  3. Across the interface, $\Gamma$, between two elements $i$ and $j$, impose an $L_2$ continuity requirement $\int_{\Gamma}(u_i-u_j)\psi d\tau = 0 \forall \psi \in \mathbb{P}_{N_e-2}(\Gamma)$
