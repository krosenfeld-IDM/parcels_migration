import sys
import numpy as np
from datetime import timedelta as delta
from pathlib import Path
import parcels

root_dir = Path(__file__).resolve().parents[1]
sys.path.append((root_dir / 'scripts').as_posix())
from example_analyticaladvection import MyParticle, radialrotation_fieldset, UpdateR, doublegyre_fieldset, bickleyjet_fieldset, ZonalBC

def test_example_1():
    pset = parcels.ParticleSet(radialrotation_fieldset(), pclass=MyParticle, lon=0, lat=4e3, time=0)

    pset.execute(
        pset.Kernel(UpdateR) + parcels.AdvectionAnalytical,
        runtime=delta(hours=24),
        dt=delta(hours=1),  # needs to be set to np.inf for Analytical Advection
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
        dt=delta(hours=1),
        runtime=delta(days=0.5),
    )

    # Like with the double gyre above, we can also compute these trajectories with the `AdvectionRK4` kernel
    psetRK4 = parcels.ParticleSet(fieldsetBJ, pclass=parcels.JITParticle, lon=X, lat=Y)
    psetRK4.execute(
        parcels.AdvectionRK4 + psetRK4.Kernel(ZonalBC),
        dt=delta(minutes=5),
        runtime=delta(days=0.5),
    )    