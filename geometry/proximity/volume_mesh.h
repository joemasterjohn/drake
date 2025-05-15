#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <sycl/sycl.hpp>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/mesh_traits.h"
#include "drake/math/linear_solve.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
/** %VolumeElement represents a tetrahedral element in a VolumeMesh. It is a
 topological entity in the sense that it only knows the indices of its vertices
 but not their coordinates.
 */
class VolumeElement {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(VolumeElement);

  /** Constructs VolumeElement.
   We follow the convention that the first three vertices define a triangle with
   its right-handed normal pointing inwards. The fourth vertex is then on the
   positive side of this first triangle.
   @warning This class does not enforce our convention for the ordering of the
   vertices.
   @param v0 Index of the first vertex in VolumeMesh.
   @param v1 Index of the second vertex in VolumeMesh.
   @param v2 Index of the third vertex in VolumeMesh.
   @param v3 Index of the last vertex in VolumeMesh.
   @pre All indices are non-negative.
   */
  VolumeElement(int v0, int v1, int v2, int v3) : vertex_({v0, v1, v2, v3}) {
    DRAKE_DEMAND(v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0);
  }

  /** Constructs VolumeElement.
   @param v  Array of four integer indices of the vertices of the element in
             VolumeMesh.
   @pre All indices are non-negative.
   */
  explicit VolumeElement(const int v[4])
      : VolumeElement(v[0], v[1], v[2], v[3]) {}

  /** Returns the number of vertices in this element. */
  int num_vertices() const { return 4; }

  /** Returns the vertex index in VolumeMesh of the i-th vertex of this
   element.
   @param i  The local index of the vertex in this element.
   @pre 0 <= i < 4
   */
  int vertex(int i) const { return vertex_.at(i); }

  /** Checks to see whether the given VolumeElement use the same four indices in
   the same order. We check for equality to the last bit consistently with
   VolumeMesh::Equal(). Two permutations of the four vertex indices of a
   tetrahedron are considered different tetrahedra even though they span the
   same space.
   */
  bool Equal(const VolumeElement& e) const {
    return this->vertex_ == e.vertex_;
  }

 private:
  // The vertices of this element.
  std::array<int, 4> vertex_;
};

inline bool operator==(const VolumeElement& e1, const VolumeElement& e2) {
  return e1.Equal(e2);
}

inline bool operator!=(const VolumeElement& e1, const VolumeElement& e2) {
  return !(e1 == e2);
}

// Forward declaration of VolumeMeshTester<T>. VolumeMesh<T> will grant
// friend access to VolumeMeshTester<T>.
template <typename T>
class VolumeMeshTester;

/** %VolumeMesh represents a tetrahedral volume mesh.
 @tparam T  The underlying scalar type for coordinates, e.g., double or
            AutoDiffXd. Must be a valid Eigen scalar.
 */
template <class T, typename Alloc = std::allocator<void>>
class VolumeMesh {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(VolumeMesh);

  /**
   @name Mesh type traits

   A collection of type traits to enable mesh consumers to be templated on
   mesh type. Each mesh type provides specific definitions of _vertex_,
   _element_, and _barycentric coordinates_. For %VolumeMesh, an element is a
   tetrahedron.
   */
  //@{

  using ScalarType = T;

  // Rebind the allocator to a new type U
  template <typename U>
  using RebindAlloc =
      typename std::allocator_traits<Alloc>::template rebind_alloc<U>;

  // Define a vector with the rebound allocator
  template <typename U>
  using vector_type = std::vector<U, RebindAlloc<U>>;

  /**
   Number of vertices per element. A tetrahedron has 4 vertices.
   */
  static constexpr int kVertexPerElement = 4;

  // TODO(SeanCurtis-TRI) This is very dissatisfying. The alias contained in a
  //  templated class doesn't depend on the class template parameter, but
  //  depends on some non-template-dependent property (kVertexPerElement).
  //  That means we *apparently* have different types:
  //    VolumeMesh<double>::Barycentric<double>
  //    VolumeMesh<AutoDiffXd>::Barycentric<double>
  // But, ultimately both become Vector4d and, because they are simply aliases,
  // are interchangeable. It would be nice to have some way of formulating this
  // that *doesn't* imply dependency on the scalar type of VolumeMesh.
  /** Type of barycentric coordinates on a tetrahedral element. Barycentric
   coordinates (b₀, b₁, b₂, b₃) satisfy b₀ + b₁ + b₂ + b₃ = 1. It corresponds
   to a position in the space. If all bᵢ >= 0, it corresponds to a position
   inside the tetrahedron or on the faces of the tetrahedron. If some bᵢ < 0,
   it corresponds to a position outside the tetrahedron. Technically we
   could calculate one of the bᵢ from the others; however, there is no
   standard way to omit one of the coordinates.
  */
  template <typename U = T>
  using Barycentric = Vector<U, kVertexPerElement>;

  //@}

  /** Constructor from a vector of vertices and from a vector of elements.
   Each element must be a valid VolumeElement following the vertex ordering
   convention documented in the VolumeElement class. This class however does not
   enforce this convention and it is thus the responsibility of the user.  */
  VolumeMesh(vector_type<VolumeElement>&& elements,
             vector_type<Vector3<T>>&& vertices);

  const VolumeElement& element(int e) const {
    DRAKE_DEMAND(0 <= e && e < num_elements());
    return elements_[e];
  }

  /** Returns the vertex identified by a given index.
   @param v  The index of the vertex.
   @pre v ∈ {0, 1, 2,...,num_vertices()-1}.
   */
  const Vector3<T>& vertex(int v) const {
    DRAKE_DEMAND(0 <= v && v < num_vertices());
    return vertices_M_[v];
  }

  /** Returns the inward facing normal of face f of element e.
   @param e The index of the element.
   @param f The index of the triangular face of the tetrahedral element e
            formed by the vertices [(f + 1) % 4, (f + 2) % 4, (f + 3) % 4].
   @pre e ∈ [0, num_elements())
   @pre f ∈ [0, 4)
   */
  const Vector3<T>& inward_normal(int e, int f) const {
    DRAKE_DEMAND(0 <= e && e < num_elements());
    DRAKE_DEMAND(0 <= f && f < kVertexPerElement);
    return inward_normals_M_[e][f];
  }

  /** Returns p_AB_M, the position vector from vertex A to vertex B in M, where
   A and B are specified by the element local indices a and b of element e.
   @param e The index of the element.
   @param a The element local index of vertex A.
   @param b The element local index of vertex B.
   @pre e ∈ [0, num_elements())
   @pre a ∈ [0, 4)
   @pre b ∈ [0, 4)
   @pre a < b
  */
  const Vector3<T>& edge_vector(int e, int a, int b) const {
    DRAKE_DEMAND(0 <= e && e < num_elements());
    DRAKE_DEMAND(0 <= a && a < kVertexPerElement);
    DRAKE_DEMAND(0 <= b && b < kVertexPerElement);
    DRAKE_DEMAND(a < b);
    // The following formula gives this table:
    // {a, b} = {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    // index  =      0,      1,      2,      3,      4,      5
    const int index = a + b - !a;
    return edge_vectors_M_[e][index];
  }

  const vector_type<Vector3<T>>& vertices() const { return vertices_M_; }

  const vector_type<VolumeElement>& tetrahedra() const { return elements_; }

  /** Returns the number of tetrahedral elements in the mesh.
   */
  int num_elements() const { return elements_.size(); }

  /** Returns the number of vertices in the mesh.
   */
  int num_vertices() const { return vertices_M_.size(); }

  /** Calculates volume of a tetrahedral element. It is a signed volume, i.e.,
   it can be negative depending on the order of the four vertices of the
   tetrahedron.
   @pre `e ∈ [0, num_elements())`.
   */
  T CalcTetrahedronVolume(int e) const {
    const T volume =
        (this->edge_vector(e, 0, 3).dot(
            this->edge_vector(e, 0, 1).cross(this->edge_vector(e, 0, 2)))) /
        T(6.0);
    return volume;
  }

  /** Calculates the volume of `this` mesh by taking the sum of the volume of
   *  each tetrahedral element.
   */
  T CalcVolume() const {
    T volume(0.0);
    for (int e = 0; e < num_elements(); ++e) {
      volume += CalcTetrahedronVolume(e);
    }
    return volume;
  }

  /** Calculate barycentric coordinates with respect to the tetrahedron `e`
   of the point Q. This operation is expensive compared with going from
   barycentric to Cartesian.

   The return type depends on both the mesh's vertex position scalar type `T`
   and the Cartesian coordinate type `C` of the query point.  See
   @ref drake::geometry::promoted_numerical "promoted_numerical_t" for details.

   @param p_MQ  A position expressed in the frame M of the mesh.
   @param e     The index of a tetrahedral element.
   @note  If p_MQ is outside the tetrahedral element, the barycentric
          coordinates (b₀, b₁, b₂, b₃) still satisfy b₀ + b₁ + b₂ + b₃ = 1;
          however, some bᵢ will be negative.
   */
  template <typename C>
  Barycentric<promoted_numerical_t<T, C>> CalcBarycentric(
      const Vector3<C>& p_MQ, int e) const {
    // We have two conditions to satisfy.
    // 1. b₀ + b₁ + b₂ + b₃ = 1
    // 2. b₀*v0 + b₁*v1 + b₂*v2 + b₃*v3 = p_M.
    // Together they create this 4x4 linear system:
    //
    //      | 1  1  1  1 ||b₀|   | 1 |
    //      | |  |  |  | ||b₁| = | | |
    //      | v0 v1 v2 v3||b₂|   |p_M|
    //      | |  |  |  | ||b₃|   | | |
    //
    // q = p_M - v0 = b₀*u0 + b₁*u1 + b₂*u2 + b₃*u3
    //              = 0 + b₁*u1 + b₂*u2 + b₃*u3
    using ReturnType = promoted_numerical_t<T, C>;
    Matrix4<ReturnType> A;
    for (int i = 0; i < 4; ++i) {
      A.col(i) << ReturnType(1.0), vertex(element(e).vertex(i));
    }
    Vector4<ReturnType> b;
    b << ReturnType(1.0), p_MQ;
    const math::LinearSolver<Eigen::PartialPivLU, Matrix4<ReturnType>> A_lu(A);
    const Vector4<ReturnType> b_Q = A_lu.Solve(b);
    // TODO(DamrongGuoy): Save the inverse of the matrix instead of
    //  calculating it on the fly. We can reduce to 3x3 system too.  See
    //  issue #11653.
    return b_Q;
  }

  /** Checks to see whether the given VolumeMesh object is equal via deep
   comparison (up to a tolerance). NaNs are treated as not equal as per the IEEE
   standard. The tolerance is applied to corresponding vertex positions; the ith
   vertex in each mesh can have a distance of no more than `vertex_tolerance`.
   @param mesh              The mesh for comparison.
   @param vertex_tolerance  The maximum distance allowed between two vertices to
                            be considered equal.
   @returns `true` if the given mesh is equal.
   */
  bool Equal(const VolumeMesh<T, Alloc>& mesh,
             double vertex_tolerance = 0) const;

  /** Calculates the gradient ∇u of a linear field u on the tetrahedron `e`.
   Field u is defined by the four field values `field_value[i]` at the i-th
   vertex of the tetrahedron. The gradient ∇u is expressed in the coordinates
   frame of this mesh M.

   If the return value is std::nullopt, the tetrahedron is degenerate, and no
   reliable gradient could be computed.

   The return type depends on both the mesh's vertex position scalar type `T`
   and the given field's scalar type `FieldValue`.  See
   @ref drake::geometry::promoted_numerical "promoted_numerical_t" for details.
   */
  template <typename FieldValue>
  std::optional<Vector3<promoted_numerical_t<T, FieldValue>>>
  MaybeCalcGradientVectorOfLinearField(
      const std::array<FieldValue, 4>& field_value, int e) const {
    using ReturnType = promoted_numerical_t<T, FieldValue>;
    Vector3<ReturnType> gradu_M = Vector3<ReturnType>::Zero();
    for (int i = 0; i < 4; ++i) {
      auto grad_i = MaybeCalcGradBarycentric(e, i);
      if (!grad_i.has_value()) {
        return {};
      }
      gradu_M += field_value[i] * *grad_i;
    }
    return gradu_M;
  }

  /** Like MaybeCalcGradientVectorOfLinearField, but throws if the geometry is
   degenerate.

   @throws std::exception if the gradient could not be computed.
   */
  template <typename FieldValue>
  Vector3<promoted_numerical_t<T, FieldValue>> CalcGradientVectorOfLinearField(
      const std::array<FieldValue, 4>& field_value, int e) const {
    auto result = MaybeCalcGradientVectorOfLinearField(field_value, e);
    if (!result.has_value()) {
      throw std::runtime_error("Bad geometry; could not calculate gradient.");
    }
    return result.value();
  }

  /** Transforms the vertices of this mesh from its initial frame M to the new
   frame N.
   @param[in] transform  The transform X_NM relating the mesh in frame M to the
   new frame N. */
  void TransformVertices(const math::RigidTransform<T>& transform);

  /** Updates the position of all vertices in the mesh. Each sequential triple
   in p_MVs (e.g., 3i, 3i + 1, 3i + 2), i ∈ ℤ, is interpreted as a position
   vector associated with the iᵗʰ vertex. The position values are interpreted to
   be measured and expressed in the same frame as the mesh to be deformed.

   @param p_MVs  Vertex positions for the mesh's N vertices flattened into a
                 vector (where each position vector is measured and expressed in
                 the mesh's original frame).
   @throws std::exception if p_MVs.size() != 3 * num_vertices() */
  void SetAllPositions(const Eigen::Ref<const VectorX<T>>& p_MVs);

 private:
  // Calculates the gradient vector ∇bᵢ of the barycentric coordinate
  // function bᵢ of the i-th vertex of the tetrahedron `e`. The gradient
  // vector ∇bᵢ is expressed in the coordinates frame of this mesh M.
  // @pre  0 ≤ i < 4.
  // TODO(rpoyner-tri): currently only used by the test helper; delete?
  Vector3<T> CalcGradBarycentric(int e, int i) const {
    auto result = MaybeCalcGradBarycentric(e, i);
    if (!result.has_value()) {
      throw std::runtime_error("Bad geometry; could not calculate gradient.");
    }
    return *result;
  }

  void TransformVerticesImpl(const math::RigidTransform<T>& transform);

  // Calculates the inward facing normals and element edge vectors.
  void ComputePositionDependentQuantities();

  // Like CalcGradBarycentric, but returns std::nullopt instead of throwing on
  // degenerate geometry.
  std::optional<Vector3<T>> MaybeCalcGradBarycentric(int e, int i) const;

  // The tetrahedral elements that comprise the volume.
  vector_type<VolumeElement> elements_;
  // The vertices that are shared between the tetrahedral elements, measured and
  // expressed in the mesh's frame M.
  vector_type<Vector3<T>> vertices_M_;
  // Stores the inward facing normals of each face of the tetrahedron, measured
  // and expressed in the mesh's frame M. Index i stores the normal of the face
  // formed by vertices {0, 1, 2, 3} / {i}.
  vector_type<std::array<Vector3<T>, 4>> inward_normals_M_;
  // Stores the edge vectors of each tetrahedron, measured and
  // expressed in the mesh's frame M, in lexicographical order:
  // {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
  vector_type<std::array<Vector3<T>, 6>> edge_vectors_M_;

  friend class VolumeMeshTester<T>;
};

template <typename Alloc>
SYCL_EXTERNAL void VolumeMesh<double, Alloc>::TransformVertices(
    const math::RigidTransform<double>& transform);

DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class VolumeMesh);

}  // namespace geometry
}  // namespace drake
