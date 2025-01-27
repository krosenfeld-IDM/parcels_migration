import parcels
import numpy as np
from datetime import timedelta


def get_test_particles(fieldset):
    return parcels.ParticleSet.from_list(
        fieldset,
        pclass=parcels.JITParticle,
        lon=np.zeros(100),
        lat=np.ones(100) * 0.75,
        time=np.zeros(100),
        lonlatdepth_dtype=np.float64,
    )


# kernel for Smagorinsky diffusion method
def smagdiff(particle, fieldset, time):
    dx = 0.01
    # gradients are computed by using a local central difference.
    updx, vpdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon + dx]
    umdx, vmdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon - dx]
    updy, vpdy = fieldset.UV[time, particle.depth, particle.lat + dx, particle.lon]
    umdy, vmdy = fieldset.UV[time, particle.depth, particle.lat - dx, particle.lon]

    dudx = (updx - umdx) / (2 * dx)
    dudy = (updy - umdy) / (2 * dx)

    dvdx = (vpdx - vmdx) / (2 * dx)
    dvdy = (vpdy - vmdy) / (2 * dx)

    A = fieldset.cell_areas[time, 0, particle.lat, particle.lon]
    sq_deg_to_sq_m = (1852 * 60) ** 2 * math.cos(particle.lat * math.pi / 180)
    A = A / sq_deg_to_sq_m
    Kh = fieldset.Cs * A * math.sqrt(dudx**2 + 0.5 * (dudy + dvdx) ** 2 + dvdy**2)

    dlat = ParcelsRandom.normalvariate(0.0, 1.0) * math.sqrt(
        2 * math.fabs(particle.dt) * Kh
    )
    dlon = ParcelsRandom.normalvariate(0.0, 1.0) * math.sqrt(
        2 * math.fabs(particle.dt) * Kh
    )

    particle_dlat += dlat
    particle_dlon += dlon


def DeleteParticle(particle, fieldset, time):
    if particle.state == parcels.StatusCode.ErrorOutOfBounds:
        particle.delete()


def smagdiff_example():
    example_dataset_folder = parcels.download_example_dataset(
        "GlobCurrent_example_data"
    )
    # read velocity files from netcdf files
    filenames = {
        "U": f"{example_dataset_folder}/20*.nc",
        "V": f"{example_dataset_folder}/20*.nc",
    }
    variables = {
        "U": "eastward_eulerian_current_velocity",
        "V": "northward_eulerian_current_velocity",
    }
    dimensions = {"lat": "lat", "lon": "lon", "time": "time"}
    fieldset = parcels.FieldSet.from_netcdf(filenames, variables, dimensions)
    # Adding parameters (`cell_areas` – areas of computational cells, and `Cs` – Smagorinsky constant) to `fieldset` that are needed for the `smagdiff` kernel
    x = fieldset.U.grid.lon
    y = fieldset.U.grid.lat
    cell_areas = parcels.Field(
        name="cell_areas", data=fieldset.U.cell_areas(), lon=x, lat=y
    )
    fieldset.add_field(cell_areas)
    fieldset.add_constant("Cs", 0.1)

    # Release particles at one location every 12 hours
    lon = 29
    lat = -33
    repeatdt = timedelta(hours=12)
    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.JITParticle,
        lon=lon,
        lat=lat,
        repeatdt=repeatdt,
    )

    kernels = pset.Kernel(parcels.AdvectionRK4) + pset.Kernel(smagdiff) + DeleteParticle
    # Modeling the particles moving during 5 days using advection (`AdvectionRK4`) and diffusion (`smagdiff`) kernels.
    pset.execute(
        kernels,
        runtime=timedelta(days=5),
        dt=timedelta(minutes=5),
    )


def basic_diffusion_example():
    # Average diffusivity
    K_bar = 0.5
    # Profile steepness
    alpha = 1.0  # Profile steepness
    # Basin scale
    L = 1.0
    # Number of grid cells in y_direction
    Ny = 103
    # y-coordinates used for setting diffusivity
    y_K = np.linspace(0.0, 1.0, 101)
    # placeholder for fraction term in K(y) formula
    beta = np.zeros(y_K.shape)

    for yi in range(len(y_K)):
        if y_K[yi] < L / 2:
            beta[yi] = y_K[yi] * np.power(L - 2 * y_K[yi], 1 / alpha)
        elif y_K[yi] >= L / 2:
            beta[yi] = (L - y_K[yi]) * np.power(2 * y_K[yi] - L, 1 / alpha)
    Kh_meridional = (
        0.1
        * (2 * (1 + alpha) * (1 + 2 * alpha))
        / (alpha**2 * np.power(L, 1 + 1 / alpha))
        * beta
    )
    Kh_meridional = np.concatenate((np.array([0]), Kh_meridional, np.array([0])))

    # Create a flat fieldset
    _, ydim = (1, Ny)
    data = {
        "U": np.zeros(ydim),
        "V": np.zeros(ydim),
        "Kh_zonal": K_bar * np.ones(ydim),
        "Kh_meridional": Kh_meridional,
    }
    dims = {"lon": 1, "lat": np.linspace(-0.01, 1.01, ydim, dtype=np.float32)}
    fieldset = parcels.FieldSet.from_data(
        data, dims, mesh="flat", allow_time_extrapolation=True
    )
    fieldset.add_constant("dres", 0.00005)

    # simulate the advection and diffusion of the particles using the `AdvectionDiffusionM1` kernel.
    dt = 0.001
    testParticles = get_test_particles(fieldset)
    # Random seed for reproducibility
    parcels.ParcelsRandom.seed(1636)
    testParticles.execute(
        parcels.AdvectionDiffusionM1,
        runtime=timedelta(seconds=0.3),
        dt=timedelta(seconds=dt),
        verbose_progress=True,
    )

    # execute the simulation with the `AdvectionDiffusionEM` kernel instead.
    dt = 0.001
    testParticles = get_test_particles(fieldset)
    parcels.ParcelsRandom.seed(1636)
    testParticles.execute(
        parcels.AdvectionDiffusionEM,
        runtime=timedelta(seconds=0.3),
        dt=timedelta(seconds=dt),
        verbose_progress=True,
    )


def main():
    basic_diffusion_example()
    smagdiff_example()


if __name__ == "__main__":
    main()
