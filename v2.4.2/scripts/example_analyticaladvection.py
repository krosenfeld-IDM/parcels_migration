import parcels
import numpy as np
from datetime import timedelta as delta
import matplotlib.pyplot as plt


def radialrotation_fieldset(xdim=201, ydim=201):
    # Coordinates of the test fieldset (on C-grid in m)
    a = b = 20000  # domain size
    lon = np.linspace(-a / 2, a / 2, xdim, dtype=np.float32)
    lat = np.linspace(-b / 2, b / 2, ydim, dtype=np.float32)
    dx, dy = lon[2] - lon[1], lat[2] - lat[1]

    # Define arrays R (radius), U (zonal velocity) and V (meridional velocity)
    U = np.zeros((lat.size, lon.size), dtype=np.float32)
    V = np.zeros((lat.size, lon.size), dtype=np.float32)
    R = np.zeros((lat.size, lon.size), dtype=np.float32)

    def calc_r_phi(ln, lt):
        return np.sqrt(ln**2 + lt**2), np.arctan2(ln, lt)

    omega = 2 * np.pi / delta(days=1).total_seconds()
    for i in range(lon.size):
        for j in range(lat.size):
            r, phi = calc_r_phi(lon[i], lat[j])
            R[j, i] = r
            r, phi = calc_r_phi(lon[i] - dx / 2, lat[j])
            V[j, i] = -omega * r * np.sin(phi)
            r, phi = calc_r_phi(lon[i], lat[j] - dy / 2)
            U[j, i] = omega * r * np.cos(phi)

    data = {"U": U, "V": V, "R": R}
    dimensions = {"lon": lon, "lat": lat}
    fieldset = parcels.FieldSet.from_data(data, dimensions, mesh="flat")
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"
    return fieldset


def doublegyre_fieldset(times, xdim=51, ydim=51):
    """Implemented following Froyland and Padberg (2009), 10.1016/j.physd.2009.03.002"""
    A = 0.25
    delta = 0.25
    omega = 2 * np.pi

    a, b = 2, 1  # domain size
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(0, b, ydim, dtype=np.float32)
    dx, dy = lon[2] - lon[1], lat[2] - lat[1]

    U = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)

    for i in range(lon.size):
        for j in range(lat.size):
            x1 = lon[i] - dx / 2
            x2 = lat[j] - dy / 2
            for t in range(len(times)):
                time = times[t]
                f = (
                    delta * np.sin(omega * time) * x1**2
                    + (1 - 2 * delta * np.sin(omega * time)) * x1
                )
                U[t, j, i] = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * x2)
                V[t, j, i] = (
                    np.pi
                    * A
                    * np.cos(np.pi * f)
                    * np.sin(np.pi * x2)
                    * (
                        2 * delta * np.sin(omega * time) * x1
                        + 1
                        - 2 * delta * np.sin(omega * time)
                    )
                )

    data = {"U": U, "V": V}
    dimensions = {"lon": lon, "lat": lat, "time": times}
    allow_time_extrapolation = True if len(times) == 1 else False
    fieldset = parcels.FieldSet.from_data(
        data, dimensions, mesh="flat", allow_time_extrapolation=allow_time_extrapolation
    )
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"
    return fieldset


def bickleyjet_fieldset(times, xdim=51, ydim=51):
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


def UpdateR(particle, fieldset, time):
    particle.radius = fieldset.R[time, particle.depth, particle.lat, particle.lon]


class MyParticle(parcels.ScipyParticle):
    fieldsetRR = radialrotation_fieldset()
    radius = parcels.Variable("radius", dtype=np.float32, initial=0.0)
    radius_start = parcels.Variable(
        "radius_start", dtype=np.float32, initial=fieldsetRR.R
    )


def ZonalBC(particle, fieldset, time):
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west


def main(args=None):
    fieldsetRR = radialrotation_fieldset()

    pset = parcels.ParticleSet(fieldsetRR, pclass=MyParticle, lon=0, lat=4e3, time=0)

    pset.execute(
        pset.Kernel(UpdateR) + parcels.AdvectionAnalytical,
        runtime=delta(hours=24),
        dt=np.inf,  # needs to be set to np.inf for Analytical Advection
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
        dt=np.inf,  # needs to be set to np.inf for Analytical Advection
        runtime=3,
    )

    # Now, we can also compute these trajectories with the `AdvectionRK4` kernel
    psetRK4 = parcels.ParticleSet(fieldsetDG, pclass=parcels.JITParticle, lon=X, lat=Y)
    psetRK4.execute(parcels.AdvectionRK4, dt=0.01, runtime=3)

    plt.figure()
    plt.plot(psetRK4.lon, psetRK4.lat, "r.", label="RK4")
    plt.plot(psetAA.lon, psetAA.lat, "b.", label="Analytical")
    plt.legend()
    plt.show()

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
        dt=np.inf,
        runtime=delta(days=1),
    )

    # Like with the double gyre above, we can also compute these trajectories with the `AdvectionRK4` kernel
    psetRK4 = parcels.ParticleSet(fieldsetBJ, pclass=parcels.JITParticle, lon=X, lat=Y)
    psetRK4.execute(
        parcels.AdvectionRK4 + psetRK4.Kernel(ZonalBC),
        dt=delta(minutes=5),
        runtime=delta(days=1),
    )

    # And finally, we can again compare the end locations from the `AdvectionRK4` and `AdvectionAnalytical` simulations
    plt.figure()
    plt.plot(psetRK4.lon, psetRK4.lat, "r.", label="RK4")
    plt.plot(psetAA.lon, psetAA.lat, "b.", label="Analytical")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
