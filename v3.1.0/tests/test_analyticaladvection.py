import sys
import numpy as np
from datetime import timedelta
from pathlib import Path
import parcels

root_dir = Path(__file__).resolve().parents[1]
sys.path.append((root_dir / 'scripts').as_posix())
from example_analyticaladvection import MyParticle, radialrotation_fieldset, UpdateR, doublegyre_fieldset, bickleyjet_fieldset, ZonalBC


def bickleyjet_fieldset_(times, xdim=51, ydim=51):
    """Bickley Jet Field as implemented in Hadjighasem et al 2017, 10.1063/1.4982720"""
    U0 = 0.06266
    L = 1770.0
    r0 = 6371.0
    k1 = 2 * 1 / r0
    k2 = 2 * 2 / r0
    k3 = 2 * 3 / r0
    eps1 = 0.075
    eps2 = 0.4
    eps3 = 0.3
    c3 = 0.461 * U0
    c2 = 0.205 * U0
    c1 = c3 + ((np.sqrt(5) - 1) / 2.0) * (k2 / k1) * (c2 - c3)

    a, b = np.pi * r0, 7000.0  # domain size
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(-b / 2, b / 2, ydim, dtype=np.float32)
    dx, dy = lon[2] - lon[1], lat[2] - lat[1]

    U = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    P = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)

    for i in range(lon.size):
        for j in range(lat.size):
            x1 = lon[i] - dx / 2
            x2 = lat[j] - dy / 2
            for t in range(len(times)):
                time = times[t]

                f1 = eps1 * np.exp(-1j * k1 * c1 * time)
                f2 = eps2 * np.exp(-1j * k2 * c2 * time)
                f3 = eps3 * np.exp(-1j * k3 * c3 * time)
                F1 = f1 * np.exp(1j * k1 * x1)
                F2 = f2 * np.exp(1j * k2 * x1)
                F3 = f3 * np.exp(1j * k3 * x1)
                G = np.real(np.sum([F1, F2, F3]))
                G_x = np.real(np.sum([1j * k1 * F1, 1j * k2 * F2, 1j * k3 * F3]))
                U[t, j, i] = (
                    U0 / (np.cosh(x2 / L) ** 2)
                    + 2 * U0 * np.sinh(x2 / L) / (np.cosh(x2 / L) ** 3) * G
                )
                V[t, j, i] = U0 * L * (1.0 / np.cosh(x2 / L)) ** 2 * G_x

    data = {"U": U, "V": V, "P": P}
    dimensions = {"lon": lon, "lat": lat, "time": times}
    allow_time_extrapolation = True if len(times) == 1 else False
    fieldset = parcels.FieldSet.from_data(
        data, dimensions, mesh="flat", allow_time_extrapolation=allow_time_extrapolation
    )
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"
    return fieldset

def test_example_1():
    pset = parcels.ParticleSet(radialrotation_fieldset(), pclass=MyParticle, lon=0, lat=4e3, time=0)

    pset.execute(
        pset.Kernel(UpdateR) + parcels.AdvectionAnalytical,
        runtime=timedelta(hours=24),
        dt=timedelta(hours=1),  # needs to be set to np.inf for Analytical Advection
    )

    print(f"Particle radius at start of run {pset.radius_start[0]}")
    print(f"Particle radius at end of run {pset.radius[0]}")
    print(f"Change in Particle radius {pset.radius[0] - pset.radius_start[0]}")

    fieldsetDG = doublegyre_fieldset(times=np.arange(0, 3.1, 0.1))

    # Now simulate a set of particles on this fieldset, using the `AdvectionAnalytical` kernel
    X, Y = np.meshgrid(np.arange(0.15, 1.85, 0.1), np.arange(0.15, 0.85, 0.1))
    psetAA = parcels.ParticleSet(fieldsetDG, pclass=parcels.ScipyParticle, lon=X, lat=Y)

    psetAA.execute(
        parcels.AdvectionAnalytical,
        dt=0.1,
        runtime=1,
    )

def test_example_2():

    fieldsetBJ = bickleyjet_fieldset(times=np.arange(0, 1.1, 0.1) * 86400)

    # Add a zonal halo for periodic boundary conditions in the zonal direction
    fieldsetBJ.add_constant("halo_west", fieldsetBJ.U.grid.lon[0])
    fieldsetBJ.add_constant("halo_east", fieldsetBJ.U.grid.lon[-1])
    fieldsetBJ.add_periodic_halo(zonal=True)

    # And simulate a set of particles on this fieldset, using the `AdvectionAnalytical` kernel
    X, Y = np.meshgrid(np.arange(0, 19900, 100), np.arange(-100, 100, 100))

    psetAA = parcels.ParticleSet(
        fieldsetBJ, pclass=parcels.ScipyParticle, lon=X, lat=Y, time=0
    )

    psetAA.execute(
        parcels.AdvectionAnalytical + psetAA.Kernel(ZonalBC),
        dt=timedelta(hours=1),
        runtime=timedelta(days=0.5),
    )

    # Like with the double gyre above, we can also compute these trajectories with the `AdvectionRK4` kernel
    psetRK4 = parcels.ParticleSet(fieldsetBJ, pclass=parcels.JITParticle, lon=X, lat=Y)
    psetRK4.execute(
        parcels.AdvectionRK4 + psetRK4.Kernel(ZonalBC),
        dt=timedelta(minutes=5),
        runtime=timedelta(days=0.5),
    )

def test_ZonalBC():

    fieldsetBJ = bickleyjet_fieldset_(times=np.arange(0, 1.1, 0.1) * 86400)

    # Add a zonal halo for periodic boundary conditions in the zonal direction
    fieldsetBJ.add_constant("halo_west", fieldsetBJ.U.grid.lon[0])
    fieldsetBJ.add_constant("halo_east", fieldsetBJ.U.grid.lon[-1])
    fieldsetBJ.add_periodic_halo(zonal=True)

    # And simulate a set of particles on this fieldset, using the `AdvectionAnalytical` kernel
    X, Y = np.meshgrid(np.arange(0, 19900, 100), np.arange(-100, 100, 100))

    psetAA = parcels.ParticleSet(
        fieldsetBJ, pclass=parcels.ScipyParticle, lon=X, lat=Y, time=0
    )

    psetAA.execute(
        parcels.AdvectionAnalytical + psetAA.Kernel(ZonalBC),
        dt=timedelta(hours=1),
        runtime=timedelta(days=0.5),
    )

    # Like with the double gyre above, we can also compute these trajectories with the `AdvectionRK4` kernel
    psetRK4 = parcels.ParticleSet(fieldsetBJ, pclass=parcels.JITParticle, lon=X, lat=Y)
    psetRK4.execute(
        parcels.AdvectionRK4 + psetRK4.Kernel(ZonalBC),
        dt=timedelta(minutes=5),
        runtime=timedelta(days=0.5),
    )