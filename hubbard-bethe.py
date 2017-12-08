#!/usr/bin/env python
import numpy
from matplotlib import pyplot


def solve(q, b, u, n_k=100, n_lambda=100):
    """
    Solves Bethe ansatz for given parameters Q, B, see Lieb, Wu, Physica A: Statistical Mechanics and its Applications,
    vol. 321, no. 1-2, pp. 1-27, Apr. 2003, section 4.

    Args:
        q (float): the wave number range;
        b (float): the lambda range;
        u (float): the Hubbard interaction term assuming the nearest heighbour hopping is equal to -1;
        n_k (int): number of points to discretize k-integrations;
        n_lambda (int): number of points to discretize lambda-integrations;

    Returns:
        A dict with all calculated quantities.
    """

    def k_func(x):
        """The K function, Eq. 23"""
        return 8 * u / (u ** 2 + 16 * x ** 2) / 2 / numpy.pi

    def k2_func(x):
        """The K-squared function, Eq. 23. Note that is is not the same as the previous function squared."""
        return 4 * u / (u ** 2 + 4 * x ** 2) / 2 / numpy.pi

    # Discretize k and l(ambda)
    k = numpy.linspace(-q, q, n_k, endpoint=False)
    dk = k[1]-k[0]

    l = numpy.linspace(-b, b, n_lambda)
    dl = l[1]-l[0]

    # Compose the linear system
    # External variables k_i, l_i and integration parameters k_j, l_j
    k_i = k[:, numpy.newaxis]
    k_j = k[numpy.newaxis, :]
    l_i = l[:, numpy.newaxis]
    l_j = l[numpy.newaxis, :]

    # Calculate discretized integrands and right-hand sides as a function of indexes i,j
    # Eq. 22
    integrand11 = numpy.eye(len(k))
    integrand12 = - numpy.cos(k_i) * k_func(numpy.sin(k_i) - l_j) * dl
    rhs1 = numpy.ones(len(k)) / 2 / numpy.pi

    # Eq. 23
    integrand21 = -k_func(numpy.sin(k_j) - l_i) * dk
    integrand22 = numpy.eye(len(l)) + k2_func(l_i - l_j) * dl
    rhs2 = numpy.zeros(len(l))

    # Compose the linear system and solve it
    A = numpy.block([[integrand11, integrand12], [integrand21, integrand22]])
    b = numpy.concatenate((rhs1, rhs2))
    x = numpy.linalg.solve(A, b)

    # Decompose the solution into unknowns rho, sigma
    rho = x[:len(k)]
    sigma = x[len(k):]

    # Calculate quantities
    # Eq. 19
    particles_per_site = sum(rho)*dk
    spin_downs_per_site = sum(sigma)*dl
    spin_ups_per_site = particles_per_site - spin_downs_per_site
    magnetization = (spin_ups_per_site - spin_downs_per_site) / particles_per_site
    # Eq. 25
    energy = - 2*sum(rho*numpy.cos(k)) * dk

    return dict(
        energy=energy,
        magnetization=magnetization,
        particles_per_site=particles_per_site,
        rho=rho,
        sigma=sigma,
        k=k,
        l=l,
    )


def batch(
        u,
        q_min=0.01,
        q_max=numpy.pi,
        nq=20,
        b_min=0,
        b_max=2,
        nb=20,
        collect=("energy", "magnetization", "particles_per_site"),
        **kwargs
):
    """
    Performs a batch calculation on a given range of parameters.
    Args:
        u (float): the Hubbard interaction term assuming the nearest heighbour hopping is equal to -1;
        q_min (float): minimal value of Q;
        q_max (float): maximal value of Q;
        nq (int): number of points to discretize Q;
        b_min (float): minimal value of B;
        b_max (float): maximal value of B;
        nb (int): number of points to discretize B;
        collect (list, tuple): a list to quantities to calculate;
        **kwargs: keyword arguments passed to `solve`.

    Returns:
        A tuple of calculated quantities as 2D arrays on the given grid of parameters.
    """
    # Set the parameter grid
    q_space = numpy.linspace(q_min, q_max, nq)
    b_space = numpy.linspace(b_min, b_max, nb)
    # Create a list of results
    results = []
    for _ in collect:
        results.append([])
    # Iterate
    for q in q_space:
        for r in results:
            r.append([])
        for b in b_space:
            result = solve(q, b, u, **kwargs)
            for r, name in zip(results, collect):
                r[-1].append(result[name])
    # Cast to numpy
    return tuple(numpy.array(r) for r in results)


# Plot ground state energy as a function of particle density and magnetization for U=1
energy, magnetization, density = batch(1)

pyplot.figure(figsize=(12, 10))
pyplot.subplot(221)
pyplot.tripcolor(density.reshape(-1), magnetization.reshape(-1), energy.reshape(-1))
pyplot.colorbar().set_label("Energy density")
pyplot.xlabel("Particle density")
pyplot.ylabel("Magnetization")
pyplot.title("Calculated data on the grid")

pyplot.subplot(222)
pyplot.tricontourf(density.reshape(-1), magnetization.reshape(-1), energy.reshape(-1), 20)
pyplot.colorbar().set_label("Energy density")
pyplot.xlabel("Particle density")
pyplot.ylabel("Magnetization")
pyplot.title("Interpolated")

# Plot ground state energy as a function of particle density for zero magnetization
# Magnetization becomes zero when the parameter B approaches infinity. Here I set it to a large enough value B=10
# The second plot displays values of magnetization for this B
pyplot.subplot(223)
ax1 = pyplot.gca()
pyplot.subplot(224)
ax2 = pyplot.gca()
for U in (1, 4, 8):
    energy, magnetization, density = batch(U, b_min=10, b_max=10, nb=1, nq=100, n_lambda=1000)
    ax1.plot(density, -energy, label="U={:d}".format(U))
    ax2.plot(density, magnetization, label="U={:d}".format(U))
ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)
ax1.set_ylim(0, 1.2)
ax1.grid(ls="--")
ax1.set_xlabel("Particle density")
ax2.set_xlabel("Particle density")
ax1.set_ylabel("Energy per site -E/t")
ax2.set_ylabel("Magnetization")
pyplot.legend()

pyplot.savefig('plot.png')
pyplot.show()
