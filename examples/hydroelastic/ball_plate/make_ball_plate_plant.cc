#include "drake/examples/hydroelastic/ball_plate/make_ball_plate_plant.h"

#include <string>
#include <utility>

#include "drake/geometry/proximity_properties.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/tree/multibody_tree_indexes.h"
#include "drake/multibody/tree/uniform_gravity_field_element.h"
#include "drake/multibody/tree/planar_joint.h"

namespace drake {
namespace examples {
namespace ball_plate {

using Eigen::Vector3d;
using geometry::AddCompliantHydroelasticProperties;
using geometry::AddContactMaterial;
using geometry::Box;
using geometry::GeometrySet;
using geometry::ProximityProperties;
using geometry::Sphere;
using math::RigidTransformd;
using math::RotationMatrixd;
using math::RollPitchYawd;
using multibody::CoulombFriction;
using multibody::MultibodyPlant;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::PlanarJoint;

namespace {

// Add tiny visual cylinders on the ±x,y,z axes of the ball to appreciate
// its rotation.
void AddTinyVisualCylinders(const RigidBody<double>& ball, double radius,
                            MultibodyPlant<double>* plant) {
  const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
  const Vector4<double> green(0.0, 1.0, 0.0, 1.0);
  const Vector4<double> blue(0.0, 0.0, 1.0, 1.0);
  const double visual_radius = 0.05 * radius;
  const geometry::Cylinder spot(visual_radius, visual_radius);
  // N.B. We do not place the cylinder's cap exactly on the sphere surface to
  // avoid visualization artifacts when the surfaces are kissing.
  const double radial_offset = radius - 0.45 * visual_radius;
  // Let S be the sphere's frame (at its center) and C be the cylinder's
  // frame (at its center). The goal is to get Cz (frame C's z axis)
  // aligned with p_SC, with Cx and Cy arbitrary.
  // @return X_SC the pose of the spot cylinder given p_SC.
  auto spot_pose = [](const Vector3<double>& p_SC) {
    return RigidTransformd(RotationMatrixd::MakeFromOneVector(p_SC, 2 /*z*/),
                           p_SC);
  };
  plant->RegisterVisualGeometry(ball, spot_pose({radial_offset, 0., 0.}), spot,
                                "sphere_x+", red);
  plant->RegisterVisualGeometry(ball, spot_pose({-radial_offset, 0., 0.}), spot,
                                "sphere_x-", red);
  plant->RegisterVisualGeometry(ball, spot_pose({0., radial_offset, 0.}), spot,
                                "sphere_y+", green);
  plant->RegisterVisualGeometry(ball, spot_pose({0., -radial_offset, 0.}), spot,
                                "sphere_y-", green);
  plant->RegisterVisualGeometry(ball, spot_pose({0., 0., radial_offset}), spot,
                                "sphere_z+", blue);
  plant->RegisterVisualGeometry(ball, spot_pose({0., 0., -radial_offset}), spot,
                                "sphere_z-", blue);
}

}  // namespace

void AddBallPlateBodies(double radius, double mass, double hydroelastic_modulus,
                        double dissipation,
                        const CoulombFriction<double>& surface_friction,
                        double resolution_hint_factor,
                        MultibodyPlant<double>* plant) {
  DRAKE_DEMAND(plant != nullptr);

  // Add the ball. Let B be the ball's frame (at its center). The ball's
  // center of mass Bcm is coincident with Bo.
  const RigidBody<double>& ball = plant->AddRigidBody(
      "Ball", SpatialInertia<double>::SolidSphereWithMass(mass, radius));
  // Set up mechanical properties of the ball.
  ProximityProperties ball_props;
  AddContactMaterial(dissipation, {} /* point stiffness */, surface_friction,
                     &ball_props);
  AddCompliantHydroelasticProperties(radius * resolution_hint_factor,
                                     hydroelastic_modulus, &ball_props);
  plant->RegisterCollisionGeometry(ball, RigidTransformd::Identity(),
                                   Sphere(radius), "collision",
                                   std::move(ball_props));
  const Vector4<double> orange(1.0, 0.55, 0.0, 0.2);
  plant->RegisterVisualGeometry(ball, RigidTransformd::Identity(),
                                Sphere(radius), "visual", orange);
  AddTinyVisualCylinders(ball, radius, plant);

  // Add the dinner plate.
  drake::multibody::Parser parser(plant);
  // parser.AddModelsFromUrl("package://drake_models/dishes/plate_8in.sdf");

  // Add the floor. Assume the frame named "Floor" is in the SDFormat file.
  parser.AddModelsFromUrl(
      "package://drake/examples/hydroelastic/ball_plate/floor.sdf");
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("Floor"),
                    RigidTransformd::Identity());

  // Gravity acting in the -z direction.
  plant->mutable_gravity_field().set_gravity_vector(Vector3d{0, 0, -9.81});
}

void AddRollingBallBodies(double radius, double mass,
                          double hydroelastic_modulus, double tile_modulus,
                          double dissipation,
                          const CoulombFriction<double>& surface_friction,
                          double resolution_hint_factor, int num_dofs,
                          MultibodyPlant<double>* plant) {
  DRAKE_DEMAND(plant != nullptr);

  ProximityProperties props;
  AddContactMaterial(dissipation, {} /* point stiffness */, surface_friction,
                     &props);
  AddCompliantHydroelasticProperties(radius * resolution_hint_factor,
                                     hydroelastic_modulus, &props);

  // Add the ball. Let B be the ball's frame (at its center). The ball's
  // center of mass Bcm is coincident with Bo.
  ProximityProperties ball_props(props);
  const RigidBody<double>& ball = plant->AddRigidBody(
      "Ball", SpatialInertia<double>::SolidSphereWithMass(mass, radius));
  // Set up mechanical properties of the ball.
  plant->RegisterCollisionGeometry(ball, RigidTransformd::Identity(),
                                   Sphere(radius), "collision",
                                   std::move(ball_props));
  const Vector4<double> orange(1.0, 0.55, 0.0, 0.8);
  plant->RegisterVisualGeometry(ball, RigidTransformd::Identity(),
                                Sphere(radius), "visual", orange);
  AddTinyVisualCylinders(ball, radius, plant);

  if (num_dofs == 3) {
    RigidTransformd X_WF(RotationMatrixd::MakeXRotation(M_PI_2));
    plant->AddJoint<PlanarJoint>("xz_planar", plant->world_body(), X_WF, ball,
                                 X_WF, Vector3d::Zero());
  }

  // Add the tiled floor.
  const double lx = 0.5;
  const double ly = 0.5;
  const double lz = 0.2;
  const int num_tiles_x = 6;
  const int num_tiles_y = 3;
  const int num_tiles_z = 2;
  Box box(lx, ly, lz);
  const Vector4<double> grey(0.5, 0.5, 0.7, 0.8);
  const RigidBody<double>& tile = plant->AddRigidBody(
      "tile", SpatialInertia<double>::SolidBoxWithMass(mass, num_tiles_x * lx,
                                                       num_tiles_y * ly, lz));
  plant->WeldFrames(
      plant->world_frame(), tile.body_frame(),
      RigidTransformd(Vector3d(0.5 * num_tiles_x * lx, 0, -0.5 * lz)));
  for (int i = 0; i < num_tiles_x; ++i) {
    for (int j = 0; j < num_tiles_y; ++j) {
      RigidTransformd X_GT(Vector3d(0.5 * lx * (2 * i + 1 - num_tiles_x),
                                    0.5 * ly * (2 * j + 1 - num_tiles_y), 0));
      ProximityProperties tile_props(props);
      tile_props.UpdateProperty(drake::geometry::internal::kHydroGroup,
                                drake::geometry::internal::kElastic, tile_modulus);
      plant->RegisterCollisionGeometry(tile, X_GT, box,
                                       fmt::format("collision_{}_{}", i, j),
                                       std::move(tile_props));
      plant->RegisterVisualGeometry(tile, X_GT, box,
                                    fmt::format("visual_{}_{}", i, j), grey);
    }
  }

  for (int i = 0; i < num_tiles_z; ++i) {
    for (int j = 0; j < num_tiles_y; ++j) {
      RigidTransformd X_GT(RollPitchYawd(0, M_PI_2, 0),
                           Vector3d(0.5 * num_tiles_x * lx + 0.5 * lz,
                                    0.5 * ly * (2 * j + 1 - num_tiles_y),
                                    0.5 * (lz + (2 * i + 1) * lx)));
      ProximityProperties tile_props(props);
      tile_props.UpdateProperty(drake::geometry::internal::kHydroGroup,
        drake::geometry::internal::kElastic, tile_modulus);
      plant->RegisterCollisionGeometry(
          tile, X_GT, box, fmt::format("collision_wall_{}_{}", i, j),
          std::move(tile_props));
      plant->RegisterVisualGeometry(
          tile, X_GT, box, fmt::format("visual_wall_{}_{}", i, j), grey);
    }
  }

  const std::pair<std::string, GeometrySet> pair = {
      "tiles", plant->CollectRegisteredGeometries({&tile})};
  plant->ExcludeCollisionGeometriesWithCollisionFilterGroupPair(pair, pair);

  // Gravity acting in the -z direction.
  plant->mutable_gravity_field().set_gravity_vector(Vector3d{0, 0, -9.81});
}

}  // namespace ball_plate
}  // namespace examples
}  // namespace drake
