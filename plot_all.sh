# Clutter
# Ensure that the data folder exists 
# Check if the directory exists, if not don't try to plot
if [ ! -d "performance_jsons_clutter_convex_Aug13" ]; then
  echo "Error: Directory performance_jsons_clutter_convex_Aug13 does not exist"
  echo "You need this folder to plot clutter results"
else
    echo "Reading clutter results from performance_jsons_clutter_convex_Aug13"
    cd performance_plotting_scripts
    python3 clutter_speedups.py  --folders performance_jsons_clutter_convex_Aug13 performance_jsons_clutter_convex_Aug13 --legends "GPU" "CPU"
    python3 clutter_barplots.py
    python3 clutter_sanity.py
    cd ..
fi

# Spatula
# Ensure that the data folder exists 
# Check if the directory exists, if not don't try to plot
if [ ! -d "performance_jsons_spatula_slip_control_scale_convex" ]; then
  echo "Error: Directory performance_jsons_spatula_slip_control_scale_convex does not exist"
  echo "You need this folder to plot spatula results"
else
    echo "Reading spatula results from performance_jsons_spatula_slip_control_scale_convex"
    cd performance_plotting_scripts
    python3 spatula_speedups.py  --folders performance_jsons_spatula_slip_control_scale_convex performance_jsons_spatula_slip_control_scale_convex --legends "GPU" "CPU"
    python3 spatula_barplots.py
    python3 spatula_sanity.py
    cd ..
fi

# Object Scaling
# Ensure that the data folder exists
# Check if the directory exists, if not don't try to plot
if [ ! -d "performance_jsons_bvh_Aug20" ]; then
  echo "Error: Directory performance_jsons_bvh_Aug20 does not exist"
  echo "You need this folder to plot object scaling results"
else
    echo "Reading object scaling results from performance_jsons_bvh_Aug20"
    cd performance_plotting_scripts
    python3 object_scaling_speedups.py  --folders performance_jsons_bvh_Aug20 performance_jsons_bvh_Aug20 --legends "GPU" "CPU"
    python3 object_scaling_barplots.py
    python3 object_scaling_sanity.py
    cd ..
fi

# Barret Hand
# Ensure that the data folder exists
# Check if the directory exists, if not don't try to plot
if [ ! -d "performance_jsons_BarretHand_convex" ] || [ ! -d "performance_jsons_BarretHand_convex_error_control" ] || [ ! -d "performance_jsons_BarretHand" ]; then
  echo "Error: Some of the directories performance_jsons_BarretHand_convex, performance_jsons_BarretHand_convex_error_control, or performance_jsons_BarretHand do not exist"
  echo "You need this folder to plot barret hand results"
else
    echo "Reading barret hand results from performance_jsons_BarretHand_convex, performance_jsons_BarretHand_convex_error_control, and performance_jsons_BarretHand"
    cd performance_plotting_scripts
    python3 Barret_barplots.py
    cd ..
fi

# Anzu
# Ensure that the data folder exists
# Check if the directory exists, if not don't try to plot
if [ ! -d "performance_jsons_anzu" ] || [ ! -d "performance_jsons_anzu_error_control" ]; then
  echo "Error: Some of the directories performance_jsons_anzu or performance_jsons_anzu_error_control do not exist"
  echo "You need this folder to plot anzu results"
else
    echo "Reading anzu results from performance_jsons_anzu and performance_jsons_anzu_error_control"
    cd performance_plotting_scripts
    python3 anzu_barplots.py
    cd ..
fi