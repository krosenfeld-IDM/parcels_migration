# # Nemo 3D tutorial
from parcels import (
    FieldSet,
    ParticleSet,
    JITParticle,
    AdvectionRK4_3D,
    download_example_dataset,
)
from glob import glob
from datetime import timedelta as delta
import matplotlib.pyplot as plt
from parcels import Field

from parcels import logger, XarrayDecodedFilter

logger.addFilter(XarrayDecodedFilter())  # Add a filter for the xarray decoding warning

example_dataset_folder = download_example_dataset("NemoNorthSeaORCA025-N006_data")
ufiles = sorted(glob(f"{example_dataset_folder}/ORCA*U.nc"))
vfiles = sorted(glob(f"{example_dataset_folder}/ORCA*V.nc"))
wfiles = sorted(glob(f"{example_dataset_folder}/ORCA*W.nc"))
mesh_mask = f"{example_dataset_folder}/coordinates.nc"

filenames = {
    "U": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": ufiles},
    "V": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": vfiles},
    "W": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": wfiles},
}

variables = {"U": "uo", "V": "vo", "W": "wo"}
dimensions = {
    "U": {"lon": "glamf", "lat": "gphif", "depth": "depthw", "time": "time_counter"},
    "V": {"lon": "glamf", "lat": "gphif", "depth": "depthw", "time": "time_counter"},
    "W": {"lon": "glamf", "lat": "gphif", "depth": "depthw", "time": "time_counter"},
}

fieldset = FieldSet.from_nemo(filenames, variables, dimensions)

pset = ParticleSet.from_line(
    fieldset=fieldset,
    pclass=JITParticle,
    size=10,
    start=(1.9, 52.5),
    finish=(3.4, 51.6),
    depth=1,
)

kernels = pset.Kernel(AdvectionRK4_3D)
pset.execute(kernels, runtime=delta(days=4), dt=delta(hours=6))


depth_level = 8
print(
    f"Level[{int(depth_level)}] depth is: [{fieldset.W.grid.depth[depth_level]:g} {fieldset.W.grid.depth[depth_level + 1]:g}]"
)
plt.figure()
pset.show(
    field=fieldset.W,
    domain={"N": 60, "S": 49, "E": 15, "W": 0},
    depth_level=depth_level,
)
plt.savefig("../figures/tutorial_nemo_3D.png")

# ## Adding other fields like cell edges
# It is quite straightforward to add other gridded data, on the same curvilinear or any other type of grid, to the fieldset. Because it is good practice to make no changes to a `FieldSet` once a `ParticleSet` has been defined in it, we redefine the fieldset and add the fields with the cell edges from the coordinates file using `FieldSet.add_field()`.


fieldset = FieldSet.from_nemo(filenames, variables, dimensions)
e1u = Field.from_netcdf(
    filenames=mesh_mask,
    variable="e1u",
    dimensions={"lon": "glamu", "lat": "gphiu"},
    interp_method="nearest",
)
e2u = Field.from_netcdf(
    filenames=mesh_mask,
    variable="e2u",
    dimensions={"lon": "glamu", "lat": "gphiu"},
    interp_method="nearest",
)
e1v = Field.from_netcdf(
    filenames=mesh_mask,
    variable="e1v",
    dimensions={"lon": "glamv", "lat": "gphiv"},
    interp_method="nearest",
)
e2v = Field.from_netcdf(
    filenames=mesh_mask,
    variable="e2v",
    dimensions={"lon": "glamv", "lat": "gphiv"},
    interp_method="nearest",
)
fieldset.add_field(e1u)
fieldset.add_field(e2u)
fieldset.add_field(e1v)
fieldset.add_field(e2v)

# %%
