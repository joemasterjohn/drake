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
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23,
         0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
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
    resolution: float,
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
        url, time_step, meshcat, E, d, resolution, use_hydroelastic, visualize
    )
    context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, context)

    plant.SetPositionsAndVelocities(plant_context, initial_state)


    simulator = Simulator(diagram, context)
    ApplySimulatorConfig(config, simulator)

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


    print(f"\nWall clock time: {wall_time}\n")
    PrintSimulatorStatistics(simulator)


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
        default=1e-3,
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
        default=1e8,
        help="Default hydroelastic modulus [Pa].",
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
        default=0.01,
        help="Default resolution hint [m].",
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
        resolution=args.resolution,
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
