import parcels
import numpy as np
from datetime import timedelta


SampleParticleInitZero = parcels.JITParticle.add_variable("temperature")


class SampleParticleOnce(parcels.JITParticle):
    temperature = parcels.Variable("temperature", initial=0, to_write="once")


def SampleT(particle, fieldset, time):
    particle.temperature = fieldset.T[time, particle.depth, particle.lat, particle.lon]


def SampleVel(particle, fieldset, time):
    u, v = fieldset.UV[particle]


def main():
    # Velocity and temperature fields
    example_dataset_folder = parcels.download_example_dataset("Peninsula_data")
    fieldset = parcels.FieldSet.from_parcels(
        f"{example_dataset_folder}/peninsula",
        extra_fields={"T": "T"},
        allow_time_extrapolation=True,
    )

    # Particle locations and initial time
    npart = 10
    lon = 3e3 * np.ones(npart)
    lat = np.linspace(3e3, 45e3, npart, dtype=np.float32)
    time = np.arange(0, npart) * timedelta(hours=2).total_seconds()

    # Setup to sample the temperature field
    pset = parcels.ParticleSet(
        fieldset=fieldset, pclass=SampleParticleInitZero, lon=lon, lat=lat, time=time
    )
    # Casting the SampleT function to a kernel.
    sample_kernel = pset.Kernel(SampleT)

    # Now sample the temperature of the particles along their trajectories
    pset.execute(
        parcels.AdvectionRK4 + sample_kernel,
        runtime=timedelta(hours=30),
        dt=timedelta(minutes=5),
    )

    # Now show how to  sample the velocity field
    pset = parcels.ParticleSet(
        fieldset=fieldset, pclass=parcels.JITParticle, lon=lon, lat=lat, time=time
    )
    pset.execute(SampleVel)


if __name__ == "__main__":
    main()
