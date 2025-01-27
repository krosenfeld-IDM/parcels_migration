import sys
import pytest
import numpy as np
from datetime import timedelta
from pathlib import Path
import parcels

root_dir = Path(__file__).resolve().parents[1]
sys.path.append((root_dir / 'scripts').as_posix())
from example_argofloats import ArgoVerticalMovement, ArgoParticle

def test_main():
    # Load the GlobCurrent data in the Agulhas region from the example_data
    example_dataset_folder = parcels.download_example_dataset(
        "GlobCurrent_example_data"
    )
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
    # uppermost layer in the hydrodynamic data
    fieldset.mindepth = fieldset.U.depth[0]

    # Initiate one Argo float in the Agulhas Current
    pset = parcels.ParticleSet(
        fieldset=fieldset, pclass=ArgoParticle, lon=[32], lat=[-31], depth=[0]
    )

    # combine Argo vertical movement kernel with built-in Advection kernel
    kernels = ArgoVerticalMovement + pset.Kernel(parcels.AdvectionRK4)

    # Create a ParticleFile object to store the output
    output_file = pset.ParticleFile(name="argo_float", outputdt=timedelta(minutes=30))

    # Now execute the kernels for 30 days, saving data every 30 minutes
    pset.execute(
        kernels,
        runtime=timedelta(days=1),
        dt=timedelta(minutes=5),
        output_file=output_file,
    )    