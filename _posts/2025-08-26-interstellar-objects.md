---
title: "Paper Implementation: Detection of ISOs using the Otautahi-Oxford population model in the LSST"
layout: post
mathjax: true
---

This is my attempt to implement the theory in this paper ["The visibility of the ¯Otautahi-Oxford interstellar object population model in LSST"](https://arxiv.org/pdf/2502.16741) by Rosemary C. Dorsey et. al. (2025). Before moving to my notes and learnings, I consider it crucial to document my motivation behind this project:

1. **Astrophysics has always been a core interest of mine**, out of many others. Learning what this field has to offer by translating theory into code was in my backlog for a long time until now. I finally managed to get myself to begin this project, which hopefully is the start of many such projects and notes.
2. **As a GNC engineer by profession, I found the concepts in this paper to be relevant to my field of work**. Here, a probabilistic orbit sampling method is developed, which allows us to predict the entry, visibiility and behaviour of ISOs (Interstellar Objects). For GNC folks, this is not only similar to trajectory planning of spacecraft but also, it's useful for anticipating the trajectory of celestial objects with good accuracy, allowing us to design the guidance and path of space probes built to study said objects.
3. Reading research papers is hard, atleast for me. I figured this is a **good exercise to improve my reading skills and patience with papers**. Training my reasoning abilities with topics like these is also a solid pathway in my opinion.

An ISO is, as the name says, a celestial visitor to our Solar Neighbourhood, hailing from galaxies farther away. They're fascinating for many reasons. For one, *they carry a wealth of information about their origins such as the formation of their origin galaxy/planet/star, their chemical and material compositions, and any interesting events happening in outer space that caused them to fly our way*. Unfortunately, they are extremely rare to spot, and much more difficult to identify. Just a month ago, 3I/ATLAS, an ISO was freshly detected. However, this is only the 3rd ISO to be found, the first two being 1I/'Oumuamua (2017) and 2I/Borisov (2019) respectively. That's a gap of 6 years! 

With the help of wide-field sky surveys, we have been able to map out a significant portion of the universe but we haven't been able to increase the sensitivies of the survey instruments. The ones in use till date (ATLAS, ZTF etc.) have a sky coverage of up to $m_r \approx 21.0-21.7$. $m_r$ is the **r-band apparent magnitude** of an object in space (how bright it *appears* to be in the red-wavelength range). The upcoming LSST is slated to bump the limit to 24.0. That's a big improvement considering $m_r$ scales ~100x with a unit in/decrement. Observing the ISOs requires what's called a **population model**. This is a mathematical model describing the statistical behaviour of such ISOs in clusters. Given the increasing amount of high-quality data we're receiving from satellites and observatories these days, the findings are that the **velocity-distribution** of the ISOs are not as simple (Gaussian) and discrete as they once seemed to be. The Otautahi-Oxford model (OO model) is the model in focus that seeks to simulate the distribution of ISOs with a great degree of accuracy, right before the LSST begins observation.

My code so far is here. I'll polish the code and drop the files in a GitHub repository later down the line:
```python
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import G, M_sun
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, Galactocentric
import matplotlib.pyplot as plt


# G * M for Sun
MU_SUN = (G * M_sun).to(u.au**3 / u.day**2)


def max_heliocentric_distance(m_r, H_r):
    """
    Calculates the maximum heliocentric distance of an object at opposition

    Args:
        m_r: Apparent magnitude in r-band
        H_r: Absolute magnitude 
    
    Returns:
        Heliocentric distance (max) in au
    """
    r_h = 0.5 * (1 + (1 + (4 * (10 ** ((m_r - H_r)/5))))**0.5)
    return r_h * u.au


def v_to_au_per_day(v):
    """Velocity in AU/day"""
    return v.to(u.au/u.day)


def A_from_vinf(v_inf):
    """Absolute value of semi-major axis in AU"""
    v = v_to_au_per_day(v_inf)  # convert to au/day
    return (MU_SUN / v**2).to(u.au)


def B_max(v_inf, r_h):
    """Max. impact parameter for orbits entering heliocentric sphere in AU"""
    r = r_h.to(u.au)
    A = A_from_vinf(v_inf)
    return np.sqrt(r**2 + 2*A*r).to(u.au)


def t_residence(v_inf, B, r_h):
    """Returns residence time in days. Zero if B>=B_max."""
    v = v_to_au_per_day(v_inf)
    r = r_h.to(u.au)
    b = B.to(u.au)
    A = A_from_vinf(v_inf)
    Bmax = B_max(v_inf, r_h)


    if np.isscalar(b.value) and b >= Bmax:
        # If somehow B is greater than B_max, then return zero
        return 0.0 * u.day
    
    root = np.sqrt(np.maximum((Bmax**2 - b**2).value, 0.0)) * u.au
    denom = np.sqrt(A**2 + b**2)
    arg = ((root + r + A) / denom).decompose().value
    arg = np.maximum(arg, 1.0)
    tres = (2.0 / v) * (root - A * np.log(arg))
    return tres.to(u.day)


if __name__ == "__main__":
    # First, we query the Gaia DR3 database 
    # to get the stellar velocity distribution
    # Parallax is assumed to be >= 5 (within 200 parsecs)
    # Unit of parallax is mas (milliarcsecond)
    vdist_query = """
    SELECT TOP 1000
        source_id, ra, dec, parallax, pmra, pmdec, radial_velocity
    FROM
        gaiadr3.gaia_source
    WHERE
        parallax IS NOT NULL
        AND pmra IS NOT NULL
        AND pmdec IS NOT NULL
        AND radial_velocity IS NOT NULL
        AND parallax >= 5
    """

    job = Gaia.launch_job(vdist_query)
    results = job.get_results()
    df = results.to_pandas()

    # We need to convert the ICRS parameters into Cartesian galactic
    # velocities in order for them to be usable in ISO orbit sampling
    coords = SkyCoord(
        ra  = df['ra'].values * u.deg,
        dec = df['dec'].values * u.deg,
        distance = (1000.0 / df['parallax'].values) * u.pc,
        pm_ra_cosdec = df['pmra'].values * u.mas/u.yr,
        pm_dec = df['pmdec'].values * u.mas/u.yr,
        radial_velocity = df['radial_velocity'].values * u.km/u.s,
        frame = "icrs" 
    )

    # Galactic coordinate system uses longitude (l)
    # and latitude (b), which are useful for later
    # Source: https://docs.astropy.org/en/stable/coordinates/index.html#transformation
    # The goal is to get the heliocentric galactic velocity
    #
    # Converting to galactocentric is better because we directly encode the velocities
    # as it's w.r.t the galactic center, and not the solar system barycenter
    galactic = coords.transform_to(Galactocentric())
    v_star = np.vstack([
        galactic.v_x.to(u.km/u.s).value,
        galactic.v_y.to(u.km/u.s).value,
        galactic.v_z.to(u.km/u.s).value,
    ]).T
    v_relative = np.linalg.norm(v_star, axis=1)     # This is V_inf

    # Empirical PDF (this is p(v_inf)). This is also mostly where the OO model
    # replaces the PDF
    n_bins = 100
    bins = np.linspace(0, 400, n_bins+1)
    hist, edges = np.histogram(v_relative, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Visualizing the histogram because why not
    plt.hist(v_relative, bins=50, density=True, alpha=0.7, color='dodgerblue')
    plt.xlabel("Speed v [km/s]")
    plt.ylabel("Probability Density p(v)")
    plt.title("Speed Distribution of Stars")
    plt.savefig("velocity_distribution.png", dpi=300)

    dv = centers[1] - centers[0]
    assert np.isclose((hist * dv).sum(), 1.0, atol=1e-3)

    pf = centers * hist
    pf /= pf.sum()  # normalized flux-weighted PDF

    n_samples = 10000

    v_inf_samples = np.random.choice(centers, size=n_samples, p=pf)
    v_samples_aus = (v_inf_samples * u.km/u.s).to(u.au/u.yr).value

    # Pulling mr = 25 and Hr = 25 from the paper. Should get ~1.6 au
    r_h = max_heliocentric_distance(25, 25) 
    Bmax_samples = B_max(v_inf_samples * u.km/u.s, r_h)

    B_samples = np.random.uniform(0, Bmax_samples) * u.au

    print(f"Sampled velocities: {v_samples_aus[:10]}")
    print(f"Sampled impact params: {B_samples[:10]}")

    n = 0.1        # From the paper --> n = 10^-1 au^-3
    flux_iso = n * v_samples_aus * B_samples.value
    F_total = np.mean(flux_iso) * 2 * np.pi

    print(f"Mean flux per unit solid angle: {F_total}")

    print("Velocity [km/s]:", v_inf_samples[0])
    print("Velocity [au/yr]:", v_samples_aus[0])

    N = n_samples
    v_kms_q = (v_inf_samples * u.km/u.s)

    Bmax_q = B_max(v_kms_q, r_h)

    u_rand = np.random.random(size=N)
    B_samples_q = (u_rand * Bmax_q.value) * u.au   

    phi_samples = np.random.uniform(0, 2.0 * np.pi, size=N)

    T = 3 * u.year
    T_days = T.to(u.day).value

    t_res_days = np.empty(N)
    for i in range(N):
        tres_q = t_residence(v_kms_q[i], B_samples_q[i], r_h)
        t_res_days[i] = tres_q.to(u.day).value

    low = -0.5 * t_res_days
    high = 0.5 * t_res_days + T_days
    tau_days = np.where(
        high > low,
        np.random.uniform(low, high),
        low
    )
    tau_samples_q = tau_days * u.day

    print("v [km/s] (first5):", v_kms_q[:5])
    print("B [AU] (first5):", B_samples_q[:5])
    print("phi [rad] (first5):", phi_samples[:5])
    print("t_res [days] (first5):", t_res_days[:5])
    print("tau [days] (first5):", tau_days[:5])
```