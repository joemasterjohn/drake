#include "drake/multibody/fem/cholmod_sparse_matrix.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/unused.h"
#include "drake/geometry/scene_graph.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {
namespace {

using Eigen::MatrixXd;
using Eigen::Vector4i;
using Eigen::VectorXd;
using geometry::GeometryId;
using geometry::GeometryInstance;
using geometry::SceneGraph;
using geometry::SceneGraphInspector;
using geometry::SourceId;
using geometry::Sphere;
using geometry::VolumeMesh;
using math::RigidTransformd;
using std::make_unique;
using std::unique_ptr;
using std::vector;

// Make an arbitrary SPD element matrix sized 12x12.
Eigen::Matrix<double, 12, 12> dummy_matrix12x12() {
  Eigen::Matrix<double, 12, 12> A;
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 12; ++j) {
      A(i, j) = 3.14 * i + 2.7 * j;
    }
  }
  Eigen::Matrix<double, 12, 12> I =
      5 * Eigen::Matrix<double, 12, 12>::Identity();
  return A * A.transpose() + I;
}

void MakeEigenSparseMatrix(const vector<Vector4i>& elements,
                           Eigen::SparseMatrix<double>* matrix) {
  const Eigen::Matrix<double, 12, 12> element_matrix = dummy_matrix12x12();
  /* Clear old values. */
  using Iterator = typename Eigen::SparseMatrix<double>::InnerIterator;
  for (int k = 0; k < matrix->outerSize(); ++k) {
    for (Iterator it(*matrix, k); it; ++it) {
      it.valueRef() = 0;
    }
  }
  /* Add in element matrices */
  for (const Vector4i& element : elements) {
    for (int a = 0; a < 4; ++a) {
      for (int i = 0; i < 3; ++i) {
        for (int b = 0; b < 4; ++b) {
          for (int j = 0; j < 3; ++j) {
            matrix->coeffRef(element(a) * 3 + i, element(b) * 3 + j) +=
                element_matrix(a * 3 + i, b * 3 + j);
          }
        }
      }
    }
  }
}

unique_ptr<GeometryInstance> make_sphere_instance(double radius = 1.0) {
  return make_unique<GeometryInstance>(RigidTransformd::Identity(),
                                       make_unique<Sphere>(radius), "sphere");
}

Eigen::SparseMatrix<double> MakeSparseTangentMatrix() {
  SceneGraph<double> scene_graph;
  SourceId s_id = scene_graph.RegisterSource();
  constexpr double kRezHint = 0.5;
  unique_ptr<GeometryInstance> geometry_instance = make_sphere_instance();
  GeometryId g_id = scene_graph.RegisterDeformableGeometry(
      s_id, scene_graph.world_frame_id(), std::move(geometry_instance),
      kRezHint);
  const SceneGraphInspector<double>& inspector = scene_graph.model_inspector();
  const VolumeMesh<double>* reference_mesh = inspector.GetReferenceMesh(g_id);
  DRAKE_DEMAND(reference_mesh != nullptr);

  vector<Vector4i> elements;
  for (const auto element : reference_mesh->tetrahedra()) {
    elements.emplace_back(element.vertex(0), element.vertex(1),
                          element.vertex(2), element.vertex(3));
  }
  Eigen::SparseMatrix<double> matrix(3 * reference_mesh->num_vertices(),
                                     3 * reference_mesh->num_vertices());
  MakeEigenSparseMatrix(elements, &matrix);
  matrix.makeCompressed();
  return matrix;
}

GTEST_TEST(CholmodSparseMatrixTest, Solve) {
  const Eigen::SparseMatrix<double> matrix = MakeSparseTangentMatrix();

  CholmodSparseMatrix cholmod_matrix(matrix);
  cholmod_matrix.Print();
  cholmod_matrix.Factor();

  const VectorXd b = VectorXd::LinSpaced(matrix.rows(), 0.0, 1.0);
  const VectorXd x = cholmod_matrix.Solve(b);

  MatrixXd dense(matrix);
  Eigen::LLT<MatrixXd> llt(dense);
  const VectorXd expected_x = llt.solve(b);
  EXPECT_TRUE(CompareMatrices(x, expected_x,
                              4.0 * std::numeric_limits<double>::epsilon()));
}

GTEST_TEST(CholmodSparseMatrixTest, CalcSchurComplement) {
  const Eigen::SparseMatrix<double> matrix = MakeSparseTangentMatrix();
  CholmodSparseMatrix cholmod_matrix(matrix);

  const MatrixXd dense(matrix);
  MatrixXd B(matrix.rows(), 2);
  B.col(0) = VectorXd::LinSpaced(matrix.rows(), 0.0, 0.01);
  B.col(1) = VectorXd::LinSpaced(matrix.rows(), 0.0, 0.02);
  MatrixXd C = MatrixXd::Zero(2, 2);
  C(0, 0) = 0.01 * B.rows();
  C(1, 1) = 0.02 * B.rows();
  cholmod_matrix.Factor();
  const MatrixX<double> S =
      cholmod_matrix.CalcSchurComplement(B, C).get_D_complement();
  Eigen::LLT<MatrixXd> llt(dense);
  const VectorXd expected_AinvB = llt.solve(B);
  const MatrixXd expected_S = C - B.transpose() * expected_AinvB;
  EXPECT_TRUE(CompareMatrices(S, expected_S, 1e-5));
}

}  // namespace
}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
