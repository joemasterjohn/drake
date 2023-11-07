#include "drake/manipulation/schunk_wsg/schunk_wsg_controller.h"

#include "drake/manipulation/schunk_wsg/schunk_wsg_constants.h"
#include "drake/manipulation/schunk_wsg/schunk_wsg_lcm.h"
#include "drake/manipulation/schunk_wsg/schunk_wsg_plain_controller.h"
#include "drake/manipulation/schunk_wsg/schunk_wsg_trajectory_generator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/matrix_gain.h"

namespace drake {
namespace manipulation {
namespace schunk_wsg {

SchunkWsgController::SchunkWsgController(double kp, double ki, double kd) {
  systems::DiagramBuilder<double> builder;

  auto wsg_trajectory_generator =
      builder.AddSystem<SchunkWsgTrajectoryGenerator>(
          kSchunkWsgNumPositions + kSchunkWsgNumVelocities,
          kSchunkWsgPositionIndex);
  const auto state_port_index = builder.ExportInput(
      wsg_trajectory_generator->get_state_input_port(), "state");

  auto command_receiver = builder.AddSystem<SchunkWsgCommandReceiver>();
  builder.ExportInput(command_receiver->GetInputPort("command_message"),
                      "command_message");

  builder.Connect(command_receiver->get_position_output_port(),
                  wsg_trajectory_generator->get_desired_position_input_port());
  builder.Connect(command_receiver->get_force_limit_output_port(),
                  wsg_trajectory_generator->get_force_limit_input_port());

  auto wsg_controller = builder.AddSystem<SchunkWsgPlainController>(
      ControlMode::kPosition, kp, ki, kd);
  builder.ConnectInput(state_port_index,
                       wsg_controller->get_input_port_estimated_state());
  builder.Connect(wsg_trajectory_generator->get_target_output_port(),
                  wsg_controller->get_input_port_desired_state());
  builder.Connect(wsg_trajectory_generator->get_max_force_output_port(),
                  wsg_controller->get_input_port_max_force());

  builder.ExportOutput(wsg_controller->get_output_port_control(), "force");
  builder.BuildInto(this);
}

SchunkWsgDesiredStateController::SchunkWsgDesiredStateController() {
  systems::DiagramBuilder<double> builder;

  auto wsg_trajectory_generator =
      builder.AddSystem<SchunkWsgTrajectoryGenerator>(
          kSchunkWsgNumPositions + kSchunkWsgNumVelocities,
          kSchunkWsgPositionIndex);
  builder.ExportInput(wsg_trajectory_generator->get_state_input_port(),
                      "state");

  auto command_receiver = builder.AddSystem<SchunkWsgCommandReceiver>();
  builder.ExportInput(command_receiver->GetInputPort("command_message"),
                      "command_message");

  builder.Connect(command_receiver->get_position_output_port(),
                  wsg_trajectory_generator->get_desired_position_input_port());
  builder.Connect(command_receiver->get_force_limit_output_port(),
                  wsg_trajectory_generator->get_force_limit_input_port());

  // Trajectory generates negative the distance between the two fingers and the
  // velocity of that quantity. This gain splits this into the state of the two
  // fingers.
  // clang-format off
  const Eigen::MatrixX<double> M =
      (Eigen::MatrixX<double>(4,2) <<  0.5,    0,
                                      -0.5,    0,
                                         0,  0.5,
                                         0, -0.5).finished();
  // clang-format on
  auto matrix_gain = builder.AddSystem<systems::MatrixGain>(M);

  builder.Connect(wsg_trajectory_generator->get_target_output_port(),
                  matrix_gain->get_input_port());

  builder.ExportOutput(matrix_gain->get_output_port(), "desired_state");

  builder.BuildInto(this);
}

}  // namespace schunk_wsg
}  // namespace manipulation
}  // namespace drake
