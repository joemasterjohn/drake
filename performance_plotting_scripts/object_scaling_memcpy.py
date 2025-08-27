import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from the table
data = {
    "Total Geometries": [301, 151, 100, 61, 31, 16, 7, 4],
    "Total Faces (Avg)": [51949, 20288, 13498, 7506, 3537, 1486, 403, 115],
    "% of total kernel time": [9.8, 6.2, 5.5, 4.1, 2.3, 1.9, 1.0, 0.8]
}

# Create DataFrame
df = pd.DataFrame(data)

# Seaborn theme for talks (no grid)
sns.set_theme(style="ticks", font_scale=1.3)

# First plot: Total Faces vs % of total kernel time
plt.figure(figsize=(8, 5))

sns.lineplot(
    data=df,
    x="Total Faces (Avg)",
    y="% of total kernel time",
    color="steelblue",
    alpha=0.7,
    linewidth=2
)

sns.scatterplot(
    data=df,
    x="Total Faces (Avg)",
    y="% of total kernel time",
    s=120,
    color="steelblue"
)

plt.title("Device to Host Memcpy", fontsize=16, weight='bold')
plt.xlabel("Total Faces (Avg)", fontsize=14)
plt.ylabel("% of Total Kernel Time", fontsize=14)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.tight_layout()
plt.savefig("object_scaling_memcpy_faces.png", dpi=600)
plt.show()

# Second plot: Total Geometries vs % of total kernel time
plt.figure(figsize=(8, 5))

sns.lineplot(
    data=df,
    x="Total Geometries",
    y="% of total kernel time",
    color="steelblue",
    alpha=0.7,
    linewidth=2
)

sns.scatterplot(
    data=df,
    x="Total Geometries",
    y="% of total kernel time",
    s=120,
    color="steelblue"
)

plt.title("Device to Host Memcpy", fontsize=16, weight='bold')
plt.xlabel("Total Geometries", fontsize=14)
plt.ylabel("% of Total Kernel Time", fontsize=14)
plt.xlim(left=0)
plt.ylim(bottom=0)




# Tight layout for talks
plt.tight_layout()
plt.savefig("object_scaling_memcpy.png", dpi=600)
plt.show()




