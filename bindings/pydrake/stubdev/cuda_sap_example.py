# Now that the environment is set up, it's safe to import matplotlib, etc.
import numpy as np
from pydrake.stubdev import stubdev

sap_solve = stubdev.FullSolveSAP()
print("created sap solve obj")

