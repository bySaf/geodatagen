"""Physical conventions and explicitly chosen scenario defaults.

References: Peplinski et al. 1995 doi:10.1109/36.387598 (mineral density);
Archie 1942 doi:10.2118/942054-G (conductivity law, not universal coefficients);
Giannakis et al. 2016 doi:10.1109/JSTARS.2015.2468597 (heterogeneous modelling).
None of the spatial scales or demonstration targets below is a site measurement.
"""

# g/cm³, particle density used by gprMax's Peplinski implementation. Bulk soil
# density is a different quantity; porosity = 1 - bulk_density / this value.
MINERAL_DENSITY_G_CM3 = 2.66

# Engineering clearance in the demonstration scenes; avoids clipping targets at
# side/bottom boundaries. This is not a geotechnical rule or a universal PML size.
GEOMETRY_CLEARANCE_M = 0.20
# Engineering separation from the surface and other objects, two 2-cm voxels in
# the standard examples. Configurable, not an inferred natural soil property.
OBJECT_CLEARANCE_M = 0.04

# Artificial weak dielectric target. Values describe a synthetic contrast only;
# they are not asserted to be the properties of a particular pipe or rock.
INCLUSION_EPSILON_R = 4.0
INCLUSION_CONDUCTIVITY_S_M = 0.001

# Low-loss CRIM scenario approximation: water is represented by a constant ~80
# instead of Debye dispersion. This is not a broadband measured permittivity.
WATER_EPSILON_R = 80.0
# Representative mineral-grain scenario value, not fitted to a sampled soil.
GRAIN_EPSILON_R = 4.5
# Archie coefficients are calibration parameters. a=1 is an unscaled formation
# factor; m=1.5 represents an illustrative unconsolidated medium; n=2 is the
# commonly used saturation exponent. Clay-rich soils require a different model.
ARCHIE_A, ARCHIE_M, ARCHIE_N = 1.0, 1.5, 2.0

# All remaining scales in the correlated-field generator are engineering priors
# inherited from the supplied field model. They are not numbers from Giannakis.
SURFACE_CORRELATION_M = (0.16, 0.16)
ROUGHNESS_CLIP_SD = 3.0  # truncate surface excursions at ±3 requested RMS values
INTERFACE_THICKNESS_FRACTION = 0.22  # limit each perturbation to preserve order
LOCAL_CORRELATION_RATIO = 0.45  # second, shorter correlation scale
POROSITY_MIX = (0.75, 0.66)  # squared weights sum ≈1; positive shared correlation
SATURATION_MIX = (0.85, -0.53)  # squared weights sum ≈1; chosen cross-correlation
POROSITY_LIMITS = (0.03, 0.80)  # broad numerical bounds, not a measured soil range
SATURATION_LIMITS = (0.001, 1.0)  # positive floor avoids zero-conductivity log bins
PORE_CONDUCTIVITY_LOG_SD = 0.15  # chosen lognormal variability, not fitted data
