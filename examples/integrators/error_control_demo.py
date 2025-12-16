import argparse
from pydrake.all import *
from pydrake.multibody.cenic import IcfSolverParameters
import builtins

import time
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

##
#
# Compare different integration schemes on a few toy examples.
#
##


@dataclass
class SimulationExample:
    """A little container for setting up different examples."""

    name: str
    url: str
    use_hydroelastic: bool
    initial_state: np.array
    sim_time: float



def ball_on_table():
    """A sphere is dropped on a table with some initial horizontal velocity."""
    name = "Ball on table"
    url = "package://drake/examples/integrators/ball_on_table.xml"
    use_hydroelastic = True
    # initial_state = np.array(
    #     [1.0, 0.0, 0.0, 0.0, 0.05, 0.0,  0.5,
    #      1.0, 0.0, 0.0, 0.0, 0.0, 0.05,  1.0,
    #      1.0, 0.0, 0.0, 0.0, 0.05, 0.05, 1.5,
    #      1.0, 0.0, 0.0, 0.0, -0.05, 0.0, 2.0,
    #      1.0, 0.0, 0.0, 0.0, 0.0, -0.05, 2.5,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    # )
    # Initial state that produces impulse (at h = 4.281817161535433e-05)
    # initial_state = np.array(
    #     [1.0, 0.0, 0.0, 0.0, 0.3, 0.0,  0.2003100155452781,
    #      0.0, 0.0, 0.0, 0.0, 0.0, -5.826597059588274]
    # )
    # initial_state = np.array(
    #     [1.0, 0.0, 0.2,  0.0, 0.0, 0.0, 1,
    #      1.0, 0.0, -0.2, 0.01, 0.01, 0.0, 1.5,
    #      1.0, 0.2, 0.0,  0.02, 0.02, 0.0, 2,
    #      1.0, -0.2, 0.0, 0.03, 0.03, 0.0, 2.5,
    #      1.0, 0.0, 0.2,  0.04, 0.04, 0.0, 3,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # )
    # initial_state = np.array(
    #     [1.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.5,
    #      1.0, 0.2, 0.0, 0.05, 0.05, 0.0, 1,
    #      1.0, 0.0, -0.2, 0.1, 0.1, 0.0, 1.5,
    #      1.0, 0.0, 0.2, 0.15, 0.15, 0.0, 2,
    #      1.0, 0.0, 0.0, 0.2, 0.2, 0.0, 2.5,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #      0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # )
    initial_state = np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25,
         0.0, 5.0, 0.0, 0.5, 0.0, 0.0]
    )
    # # Generate non-overlapping random positions for 10 balls of radius 0.1
    # num_balls = 10
    # radius = 0.1
    # positions = []
    # while len(positions) < num_balls:
    #     candidate = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(0.5, 2.0)])
    #     if builtins.all(np.linalg.norm(candidate - p, ord=2) > 2 * radius for p in positions):
    #         positions.append(candidate)
    # # Each ball: [qw, qx, qy, qz, x, y, z] for all balls, then [vx, vy, vz, wx, wy, wz] for all balls
    # q_list = []
    # v_list = []
    # for pos in positions:
    #     # Quaternion for no rotation: [1, 0, 0, 0], position: [x, y, z]
    #     q_list.extend([1, 0, 0, 0, pos[0], pos[1], pos[2]])
    #     # Zero velocity: [vx, vy, vz, wx, wy, wz]
    #     v_list.extend([0, 0, 0, 0, 0, 0])
    # initial_state = np.array(q_list + v_list)
    sim_time = 10.0
    return SimulationExample(
        name, url, use_hydroelastic, initial_state, sim_time
    )


def clutter():
    """Several spheres fall into a box."""
    name = "Clutter"
    url = "package://drake/examples/integrators/clutter.xml"
    use_hydroelastic = True
    initial_state = np.array(
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    )
    sim_time = 3.0
    return SimulationExample(
        name, url, use_hydroelastic, initial_state, sim_time
    )


def plate_and_spatula():
    """A spatula is dropped onto a plate."""
    name = "Plate and spatula"
    url = "package://drake/examples/integrators/plate_and_spatula.sdf"
    use_hydroelastic = True
    initial_state = np.array([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02,   # plate pos
        1.0, 0.0, 0.0, 0.0, -0.05, 0.0, 0.3,  # spatula pos
        1.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.5,   # plate pos
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # plate vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # spatula vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # spatula vel
    ])
    sim_time = 2.0
    return SimulationExample(
        name, url, use_hydroelastic, initial_state, sim_time
    )

def cones():
    """Cones"""
    name = "Cones"
    url = "package://drake/examples/integrators/cones.sdf"
    use_hydroelastic = True
    initial_state = np.array([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2,   # plate pos
        1.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.6,  # spatula pos
        1.0, 0.0, 0.0, 0.0, -0.01, 0.0, 1.0,   # plate pos
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # plate vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # spatula vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # spatula vel
    ])
    sim_time = 3.0
    return SimulationExample(
        name, url, use_hydroelastic, initial_state, sim_time
    )

def teddy_and_torus():
    """A teddy bear is dropped onto a torus."""
    name = "Teddy and torus"
    url = "package://drake/examples/integrators/teddy_and_torus.sdf"
    use_hydroelastic = True
    initial_state = np.array([
        1.0, 1.0, 0.0, 0.0, -0.2, 0.0, 0.1,   # teddy pos
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1,   # teddy pos
        1.0, 1.0, 0.0, 0.0, 0.2, 0.0, 0.1,   # teddy pos
        1.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.4,  # torus pos
        1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.4,  # torus pos
        1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.4,  # torus pos
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # teddy vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # teddy vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # teddy vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # torus vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # torus vel
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # torus vel
    ])
    sim_time = 3.0
    return SimulationExample(
        name, url, use_hydroelastic, initial_state, sim_time
    )


def create_scene(
    url: str,
    time_step: float,
    meshcat: Meshcat,
    E: float,
    d: float,
    margin: float,
    barrier: float,
    resolution: float,
    hydroelastic: bool = False,
    visualize: bool = True,
):
    """
    Set up a drake system dyagram

    Args:
        xml: mjcf robot description.
        time_step: dt for MultibodyPlant.
        hydroelastic: whether to use hydroelastic contact.
        meshcat: meshcat instance for visualization.
        visualize: whether to show the visualization

    Returns:
        The system diagram, the MbP within that diagram, and the logger used to
        keep track of time steps.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=time_step
    )

    parser = Parser(plant)
    parser.AddModels(url=url)
    if time_step > 0:
        plant.set_discrete_contact_approximation(
            DiscreteContactApproximation.kLagged
        )
    plant.Finalize()

    if hydroelastic:
        sg_config = SceneGraphConfig()
        sg_config.default_proximity_properties.compliance_type = "compliant"
        sg_config.default_proximity_properties.hydroelastic_modulus = E
        sg_config.default_proximity_properties.hunt_crossley_dissipation = d
        sg_config.default_proximity_properties.margin = margin
        # sg_config.default_proximity_properties.barrier = barrier
        sg_config.default_proximity_properties.resolution_hint = resolution
        sg_config.default_proximity_properties.dynamic_friction = 0.5
        sg_config.default_proximity_properties.static_friction = 0.5
        scene_graph.set_config(sg_config)

    if visualize:
        vis_config = VisualizationConfig()
        vis_config.publish_period = 0.1
        ApplyVisualizationConfig(vis_config, builder=builder, meshcat=meshcat)

    diagram = builder.Build()
    return diagram, plant


def run_simulation(
    example: SimulationExample,
    integrator: str,
    accuracy: float,
    max_step_size: float,
    meshcat: Meshcat,
    E: float,
    d: float,
    margin: float,
    barrier: float,
    resolution: float,
    beta: float,
    visualize: bool = True,
):
    """
    Run a short simulation, and report the time-steps used throughout.

    Args:
        example: container defining the scenario to simulate.
        integrator: which integration strategy to use ("implicit_euler",
            "runge_kutta3", "cenic", "discrete").
        accuracy: the desired accuracy (ignored for "discrete").
        max_step_size: the maximum (and initial) timestep dt.
        meshcat: meshcat instance for visualization.
        visualize: whether to show stuff on meshcat (slow).

    Returns:
        Timesteps (dt) throughout the simulation, and the wall-clock time.
    """
    url = example.url
    use_hydroelastic = example.use_hydroelastic
    initial_state = example.initial_state
    sim_time = example.sim_time

    # We can use a more standard simulation setup and rely on a logger to
    # tell use the time step information. Note that in this case enabling
    # visualization messes with the time step report though.

    # Configure Drake's built-in error controlled integration
    config = SimulatorConfig()
    if integrator != "discrete":
        config.integration_scheme = integrator
    config.max_step_size = max_step_size
    config.accuracy = accuracy
    config.target_realtime_rate = 0
    config.use_error_control = True
    config.publish_every_time_step = True

    # Set up the system diagram and initial condition
    if integrator == "discrete":
        time_step = max_step_size
    else:
        time_step = 0.0
    diagram, plant = create_scene(
        url, time_step, meshcat, E, d, margin, barrier, resolution, use_hydroelastic, visualize
    )
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)

    plant.SetPositionsAndVelocities(plant_context, initial_state)

    # data_file = open('/home/joemasterjohn/tri/drake/data.txt', 'w')
    # area_file = open('/home/joemasterjohn/tri/drake/areas.txt', 'w')

    # data_file.write(f"t\th\tmin_e\tmax_e\tmean_e\tarea\tforce\t{' '.join(plant.GetStateNames())}\n")
    # prev_time = 0.0
    # areas = []
    # timesteps = []

    # def monitor(context):
    #     nonlocal prev_time
    #     nonlocal areas
    #     nonlocal timesteps
    #     sim_time = context.get_time()
    #     h = sim_time - prev_time
    #     timesteps.append(h)
    #     prev_time = sim_time

    #     plant_context = plant.GetMyContextFromRoot(context)

    #     # State vector
    #     state = plant.GetPositionsAndVelocities(plant_context)

    #     v = state[12]

    #     # Contact surfaces via QueryObject
    #     query_object = plant.get_geometry_query_input_port().Eval(plant_context)
    #     contact_surfaces = query_object.ComputeContactSurfaces(HydroelasticContactRepresentation.kPolygon)

    #     # Expect only one contact surface
    #     if len(contact_surfaces) > 0:
    #         s  = contact_surfaces[0]
    #         min_e = float('inf')
    #         max_e = -float('inf')
    #         mean_e = 0
    #         total_area = 0
    #         for face in range(s.num_faces()):
    #             e = s.poly_e_MN().EvaluateCartesian(face, s.centroid(face))
    #             A = s.area(face)
    #             min_e = min(e, min_e)
    #             max_e = max(e, max_e)
    #             mean_e += e*A
    #             total_area += A
    #             areas.append(A)

    #         mean_e /= total_area

    #         data_file.write(f"{sim_time}\t{h}\t{min_e}\t{max_e}\t{mean_e}\t{total_area}\t{' '.join(map(str, state))}\n")
    #     else:
    #         data_file.write(f"{sim_time}\t{h}\t0\t0\t0\t0\t{' '.join(map(str, state))}\n")

    #     print(f"MONITOR t: {context.get_time()}")

    #     return EventStatus.Succeeded()

    simulator = Simulator(diagram, context)
    ApplySimulatorConfig(config, simulator)
    # simulator.set_monitor(monitor)

    # if integrator == "cenic":
    #     ci = simulator.get_mutable_integrator()
    #     # We can also set some solver parameters for the integrator here
    #     params = IcfSolverParameters()
    #     params.beta = beta
    #     ci.set_solver_parameters(params)

    simulator.Initialize()

    # print(f"Running the {example.name} example with {integrator} integrator.")
    if visualize:
        input("Waiting for meshcat... [ENTER] to continue")

    # Simulate
    meshcat.StartRecording()
    start_time = time.time()
    simulator.AdvanceTo(sim_time)
    wall_time = time.time() - start_time
    meshcat.StopRecording()
    meshcat.PublishRecording()

    # area_file.write('\n'.join(map(str, areas)))

    print(f"\nWall clock time: {wall_time}\n")
    PrintSimulatorStatistics(simulator)

    x_final = plant.GetPositions(plant_context)[4]
    print(f"x_final: {x_final}\n")

    #return np.asarray(timesteps), wall_time
    return [], wall_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example",
        type=str,
        default="clutter",
        help=(
            "Which example to run. One of: gripper, ball_on_table, "
            "double_pendulum, cylinder_hydro, cylinder_point, clutter. "
        ),
    )
    parser.add_argument(
        "--integrator",
        type=str,
        default="cenic",
        help=(
            "Integrator to use, e.g., implicit_euler, runge_kutta3, cenic, "
            "discrete. Default: cenic."
        ),
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=1e-1,
        help="Integrator accuracy (ignored for discrete).",
    )
    parser.add_argument(
        "--max_step_size",
        type=float,
        default=0.1,
        help=(
            "Maximum time step size (or fixed step size for discrete "
            "integrator)."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Whether to make plots of the step size over time. Default: "
            "False."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Whether to visualize with Meshcat. Default: "
            "False."
        ),
    )
    parser.add_argument(
        "--E",
        type=float,
        default=1e9,
        help="Default hydroelastic modulus [Pa].",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1e-3,
        help="Default margin [m].",
    )
    parser.add_argument(
        "--barrier",
        type=float,
        default=1e-4,
        help="Default barrier [m].",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=10,
        help="Default H&C dissipation [s/m].",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.0025,
        help="Default resolution hint [m].",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Beta parameter for cenic integrator.",
    )


    args = parser.parse_args()

    # Set up the example system
    if args.example == "ball_on_table":
        example = ball_on_table()
    elif args.example == "clutter":
        example = clutter()
    elif args.example == "plate_and_spatula":
        example = plate_and_spatula()
    elif args.example == "cones":
        example = cones()
    elif args.example == "teddy_and_torus":
        example = teddy_and_torus()
    else:
        raise ValueError(f"Unknown example {args.example}")

    meshcat = StartMeshcat()

    time_steps, _ = run_simulation(
        example,
        args.integrator,
        args.accuracy,
        max_step_size=args.max_step_size,
        meshcat=meshcat,
        E=args.E,
        d=args.d,
        margin=args.margin,
        barrier=args.barrier,
        resolution=args.resolution,
        beta=args.beta,
        visualize=args.visualize,
    )

    if args.plot:
        times = np.cumsum(time_steps)
        plt.title(
            (
                f"{example.name} | {args.integrator} integrator | "
                f"accuracy = {args.accuracy}"
            )
        )
        plt.plot(times, time_steps, "o")
        plt.ylim(1e-10, 1e0)
        plt.yscale("log")
        plt.xlabel("time (s)")
        plt.ylabel("step size (s)")
        plt.show()
