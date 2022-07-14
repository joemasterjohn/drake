#include "drake/geometry/proximity/mesh_to_vtk.h"

#include <memory>

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/temp_directory.h"
#include "drake/common/find_resource.h"
#include "drake/geometry/proximity/make_box_field.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/make_embedded_mesh.h"
#include "drake/geometry/proximity/make_embedded_field.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/make_sphere_field.h"
#include "drake/geometry/proximity/mesh_intersection.h"
#include "drake/geometry/shape_specification.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/triangle_surface_mesh_field.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using math::RigidTransformd;
using std::unique_ptr;

// We use Box to represent all primitives in the tests when we generate
// meshes and fields. We only perform smoke tests and do not check the
// content of the output VTK files.

// TODO(DamrongGuoy): Use VTK library to read the files for verification.
//  Right now we manually load the files into ParaView for verification.

GTEST_TEST(MeshToVtkTest, BoxTetrahedra) {
  const Box box(4.0, 4.0, 2.0);
  // resolution_hint 0.5 is enough to have vertices on the medial axis.
  const auto mesh = MakeBoxVolumeMesh<double>(box, 0.5);
  WriteVolumeMeshToVtk(temp_directory() + "/" + "box_tet.vtk", mesh,
                       "Tetrahedral Mesh of Box");
}

GTEST_TEST(MeshToVtkTest, BoxTriangles) {
  const Box box(4.0, 4.0, 2.0);
  // Very coarse resolution_hint 4.0 should give the coarsest mesh.
  const auto mesh = MakeBoxSurfaceMesh<double>(box, 4.0);
  WriteSurfaceMeshToVtk(temp_directory() + "/" + "box_tri.vtk", mesh,
                        "Triangular Mesh of Box");
}

GTEST_TEST(MeshToVtkTest, BoxTetrahedraPressureField) {
  const Box box(4.0, 4.0, 2.0);
  // resolution_hint 0.5 is enough to have vertices on the medial axis.
  const auto mesh = MakeBoxVolumeMesh<double>(box, 0.5);
  const double kElasticModulus = 1.0e+5;
  const auto pressure =
      MakeBoxPressureField<double>(box, &mesh, kElasticModulus);
  WriteVolumeMeshFieldLinearToVtk(
      temp_directory() + "/" + "box_tet_pressure.vtk ", "Pressure[Pa]",
      pressure, "Pressure Field in Box");
}

// Helper function to create a contact surface between a soft box and a rigid
// box.
unique_ptr<ContactSurface<double>> BoxContactSurface() {
  const Box soft_box(4., 4., 2.);
  // resolution_hint 0.5 is enough to have vertices on the medial axis.
  const VolumeMesh<double> volume_S = MakeBoxVolumeMesh<double>(soft_box, 0.5);
  const double kElasticModulus = 1.0e+5;
  const VolumeMeshFieldLinear<double, double> field_S =
      MakeBoxPressureField<double>(soft_box, &volume_S, kElasticModulus);
  const Bvh<Obb, VolumeMesh<double>> bvh_volume_S(volume_S);
  // The soft box is at the center of World.
  RigidTransformd X_WS = RigidTransformd::Identity();

  const Box rigid_box(4, 4, 2);
  // Very coarse resolution_hint 4.0 should give the coarsest mesh.
  const TriangleSurfaceMesh<double> surface_R =
      MakeBoxSurfaceMesh<double>(rigid_box, 4.0);
  const Bvh<Obb, TriangleSurfaceMesh<double>> bvh_surface_R(surface_R);
  // The rigid box intersects the soft box in a unit cube at the corner
  // (2.0, 2.0, 1.0).
  RigidTransformd X_WR(Vector3d{3., 3., 1.});

  return ComputeContactSurfaceFromSoftVolumeRigidSurface(
      GeometryId::get_new_id(), field_S, bvh_volume_S, X_WS,
      GeometryId::get_new_id(), surface_R, bvh_surface_R, X_WR,
      HydroelasticContactRepresentation::kTriangle);
}

GTEST_TEST(MeshToVtkTest, BoxContactSurfacePressure) {
  unique_ptr<ContactSurface<double>> contact = BoxContactSurface();
  auto contact_pressure =
      dynamic_cast<const TriangleSurfaceMeshFieldLinear<double, double>*>(
          &contact->tri_e_MN());
  ASSERT_NE(contact_pressure, nullptr);
  WriteTriangleSurfaceMeshFieldLinearToVtk(
      temp_directory() + "/" + "box_rigid_soft_contact_pressure.vtk",
      "Pressure[Pa]", *contact_pressure,
      "Pressure Distribution on Contact Surface");
}

// Helper function to create a contact surface between two convex embedded mesh.
unique_ptr<ContactSurface<double>> ConvexContactSurface(
    const RigidTransformd& X_WR) {
  const std::string filename =
      FindResourceOrThrow("drake/geometry/test/convex.obj");
  Mesh mesh(filename, 1.0);

  const double kElasticModulus = 1.0e+5;
  const double kDepth = 0.02;
  const double kMargin = 0.02;
  const int kSubdivisions = 20;

  auto mesh_S = MakeEmbeddedVolumeMesh<double>(mesh, kSubdivisions, kMargin);
  auto field_S =
      MakeEmbeddedPressureField<double>(mesh, &mesh_S, kElasticModulus, kDepth);

  auto mesh_R = MakeEmbeddedVolumeMesh<double>(mesh, kSubdivisions, kMargin);
  auto field_R =
      MakeEmbeddedPressureField<double>(mesh, &mesh_R, kElasticModulus, kDepth);

  const Bvh<Obb, VolumeMesh<double>> bvh_volume_S(mesh_S);
  RigidTransformd X_WS = RigidTransformd::Identity();

  const Bvh<Obb, VolumeMesh<double>> bvh_volume_R(mesh_R);

  return IntersectCompliantVolumes<TriangleSurfaceMesh<double>,
                                   TriMeshBuilder<double>>(
      GeometryId::get_new_id(), field_S, bvh_volume_S, X_WS,
      GeometryId::get_new_id(), field_R, bvh_volume_R, X_WR);
}

GTEST_TEST(MeshToVtkTest, EmbeddedPancake) {
  const std::string filename =
      FindResourceOrThrow("drake/examples/hydroelastic/two_spheres/pancake.obj");
  Mesh mesh(filename, 1.0);

  const double kElasticModulus = 1.0e+4;
  const double kDepth = 0.01;
  const double kMargin = 0.01;
  const int kSubdivisions = 2;

  auto volume_mesh =
      MakeEmbeddedVolumeMesh<double>(mesh, kSubdivisions, kMargin);

  auto volume_field = MakeEmbeddedPressureField<double>(
      mesh, &volume_mesh, kElasticModulus, kDepth);
  WriteVolumeMeshFieldLinearToVtk("/home/joe/pancake_embedded_field.vtk",
                                  "distance", volume_field, "joe");
}

GTEST_TEST(MeshToVtkTest, EmbeddedBoxVsInternalBox) {
  const std::string filename =
      FindResourceOrThrow("drake/examples/hydroelastic/two_spheres/sphere.obj");
  Mesh mesh(filename, 1.0);

  const double kElasticModulus = 1.0e+4;
  const double kDepth = 1;
  const double kMargin = 1;
  const int kSubdivisions = 8;

  auto volume_mesh =
      MakeEmbeddedVolumeMesh<double>(mesh, kSubdivisions, kMargin);

  auto volume_field = MakeEmbeddedPressureField<double>(
      mesh, &volume_mesh, kElasticModulus, kDepth);
  WriteVolumeMeshFieldLinearToVtk("/home/joe/sphere_embedded_field.vtk",
                                  "distance", volume_field, "joe");

  Sphere s(1.0);
  auto sphere_mesh = MakeSphereVolumeMesh<double>(
      s, 0.4, TessellationStrategy::kSingleInteriorVertex);

  auto sphere_pressure =
      MakeSpherePressureField(s, &sphere_mesh, kElasticModulus);

  WriteVolumeMeshFieldLinearToVtk("/home/joe/sphere_internal_field.vtk",
                                  "distance", sphere_pressure, "joe");
}

GTEST_TEST(MeshToVtkTest, EmbeddedMesh) {
  const std::string filename =
      FindResourceOrThrow("drake/geometry/test/convex.obj");
  Mesh mesh(filename, 1.0);

  const double kElasticModulus = 1.0e+5;
  const double kDepth = 0.02;
  const double kMargin = 0.02;
  const int kSubdivisions = 20;

  auto volume_mesh =
      MakeEmbeddedVolumeMesh<double>(mesh, kSubdivisions, kMargin);

  auto volume_field = MakeEmbeddedPressureField<double>(
      mesh, &volume_mesh, kElasticModulus, kDepth);
  WriteVolumeMeshFieldLinearToVtk("/home/joe/embedded_field.vtk", "distance",
                                  volume_field, "joe");

  {
    unique_ptr<ContactSurface<double>> contact =
        ConvexContactSurface(RigidTransformd(Vector3d(0, 0.1, 0.1)));
    auto contact_pressure =
        dynamic_cast<const TriangleSurfaceMeshFieldLinear<double, double>*>(
            &contact->tri_e_MN());
    WriteTriangleSurfaceMeshFieldLinearToVtk(
        "/home/joe/convex_contact_pressure_1.vtk", "Pressure[Pa]",
        *contact_pressure, "Pressure Distribution on Contact Surface");
  }

  {
    unique_ptr<ContactSurface<double>> contact =
        ConvexContactSurface(RigidTransformd(Vector3d(0, 0.08, 0.08)));
    auto contact_pressure =
        dynamic_cast<const TriangleSurfaceMeshFieldLinear<double, double>*>(
            &contact->tri_e_MN());
    WriteTriangleSurfaceMeshFieldLinearToVtk(
        "/home/joe/convex_contact_pressure_2.vtk", "Pressure[Pa]",
        *contact_pressure, "Pressure Distribution on Contact Surface");
  }

  {
    unique_ptr<ContactSurface<double>> contact =
        ConvexContactSurface(RigidTransformd(Vector3d(0, 0.06, 0.06)));
    auto contact_pressure =
        dynamic_cast<const TriangleSurfaceMeshFieldLinear<double, double>*>(
            &contact->tri_e_MN());
    WriteTriangleSurfaceMeshFieldLinearToVtk(
        "/home/joe/convex_contact_pressure_3.vtk", "Pressure[Pa]",
        *contact_pressure, "Pressure Distribution on Contact Surface");
  }

  {
    unique_ptr<ContactSurface<double>> contact =
        ConvexContactSurface(RigidTransformd(Vector3d(0, 0.04, 0.04)));
    auto contact_pressure =
        dynamic_cast<const TriangleSurfaceMeshFieldLinear<double, double>*>(
            &contact->tri_e_MN());
    WriteTriangleSurfaceMeshFieldLinearToVtk(
        "/home/joe/convex_contact_pressure_4.vtk", "Pressure[Pa]",
        *contact_pressure, "Pressure Distribution on Contact Surface");
  }
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
