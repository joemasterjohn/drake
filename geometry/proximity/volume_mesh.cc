#include "drake/geometry/proximity/volume_mesh.h"

#include "drake/common/default_scalars.h"

namespace drake {
namespace geometry {

template <typename T, typename Alloc>
VolumeMesh<T, Alloc>::VolumeMesh(
    typename VolumeMesh<T, Alloc>::template vector_type<VolumeElement>&&
        elements,
    typename VolumeMesh<T, Alloc>::template vector_type<Vector3<T>>&& vertices)
    : elements_(std::move(elements)), vertices_M_(std::move(vertices)) {
  if (elements_.empty()) {
    throw std::logic_error("A mesh must contain at least one tetrahedron");
  }
  // ComputePositionDependentQuantities();
}

template <typename T, typename Alloc>
void VolumeMesh<T, Alloc>::TransformVerticesImpl(
    const math::RigidTransform<T>& transform) {
  const math::RigidTransform<T>& X_NM = transform;
  for (Vector3<T>& vertex : vertices_M_) {
    const Vector3<T> p_MV = vertex;
    vertex = X_NM * p_MV;
  }

  // // Transform all position dependent quantities.
  // const math::RotationMatrix<T>& R_NM = X_NM.rotation();
  // for (int i = 0; i < num_elements(); ++i) {
  //   for (int j = 0; j < 4; ++j) {
  //     inward_normals_M_[i][j] = R_NM * inward_normals_M_[i][j];
  //   }

  //   for (int j = 0; j < 6; ++j) {
  //     edge_vectors_M_[i][j] = R_NM * edge_vectors_M_[i][j];
  //   }
  // }
}

template <typename T, typename Alloc>
void VolumeMesh<T, Alloc>::TransformVertices(
    const math::RigidTransform<T>& transform) {
  TransformVerticesImpl(transform);
}

template <typename Alloc>
SYCL_EXTERNAL void VolumeMesh<double, Alloc>::TransformVertices(
    const math::RigidTransform<double>& transform) {
  TransformVerticesImpl(transform);
}

template <typename T, typename Alloc>
void VolumeMesh<T, Alloc>::SetAllPositions(
    const Eigen::Ref<const VectorX<T>>& p_MVs) {
  if (p_MVs.size() != 3 * num_vertices()) {
    throw std::runtime_error(
        fmt::format("SetAllPositions(): Attempting to deform a mesh with {} "
                    "vertices with data for {} DoFs",
                    num_vertices(), p_MVs.size()));
  }
  for (int v = 0, i = 0; v < num_vertices(); ++v, i += 3) {
    vertices_M_[v] = Vector3<T>(p_MVs[i], p_MVs[i + 1], p_MVs[i + 2]);
  }

  ComputePositionDependentQuantities();
}

template <typename T, typename Alloc>
void VolumeMesh<T, Alloc>::ComputePositionDependentQuantities() {
  inward_normals_M_.clear();
  edge_vectors_M_.clear();
  inward_normals_M_.reserve(num_elements());
  edge_vectors_M_.reserve(num_elements());
  for (int e = 0; e < num_elements(); ++e) {
    const Vector3<T>& a = vertices_M_[elements_[e].vertex(0)];
    const Vector3<T>& b = vertices_M_[elements_[e].vertex(1)];
    const Vector3<T>& c = vertices_M_[elements_[e].vertex(2)];
    const Vector3<T>& d = vertices_M_[elements_[e].vertex(3)];

    const Vector3<T> ab = b - a;
    const Vector3<T> ac = c - a;
    const Vector3<T> ad = d - a;
    const Vector3<T> bc = c - b;
    const Vector3<T> bd = d - b;
    const Vector3<T> cd = d - c;

    edge_vectors_M_.push_back(
        std::array<Vector3<T>, 6>{ab, ac, ad, bc, bd, cd});

    // Assume the first three vertices a, b, c define a triangle with its
    // right-handed normal pointing towards the inside of the tetrahedra. The
    // fourth vertex, d, is on the positive side of the plane defined by a,
    // b, c. The faces that wind CCW from inside the element are:
    //  {b d c}  Across from vertex a
    //  {a c d}  Across from vertex b
    //  {a d b}  Across from vertex c
    //  {a b c}  Across from vertex d
    //
    // For example, a standard tetrahedron looks like this:
    //
    //              Mz
    //              ┆
    //            d ●
    //              ┆
    //              ┆    c
    //            a ●┄┄┄●┄┄┄ My
    //             ╱
    //          b ●
    //          ╱
    //
    inward_normals_M_.push_back(std::array<Vector3<T>, 4>{
        bd.cross(bc).normalized(), (ac).cross(ad).normalized(),
        (ad).cross(ab).normalized(), (ab).cross(ac).normalized()});
  }
}

template <typename T, typename Alloc>
bool VolumeMesh<T, Alloc>::Equal(const VolumeMesh<T, Alloc>& mesh,
                                 double vertex_tolerance) const {
  if (this == &mesh) return true;

  if (this->num_elements() != mesh.num_elements()) return false;
  if (this->num_vertices() != mesh.num_vertices()) return false;

  // Check tetrahedral elements.
  for (int i = 0; i < this->num_elements(); ++i) {
    if (!this->element(i).Equal(mesh.element(i))) return false;
  }
  // Check vertices.
  for (int i = 0; i < this->num_vertices(); ++i) {
    if ((this->vertex(i) - mesh.vertex(i)).norm() > vertex_tolerance) {
      return false;
    }
  }

  // All checks passed.
  return true;
}

template <typename T, typename Alloc>
std::optional<Vector3<T>> VolumeMesh<T, Alloc>::MaybeCalcGradBarycentric(
    int e, int i) const {
  DRAKE_DEMAND(0 <= i && i < 4);
  // Vertex V corresponds to bᵢ in the barycentric coordinate in the
  // tetrahedron indexed by `e`.  A, B, and C are the remaining vertices of
  // the tetrahedron. Their positions are expressed in frame M of the mesh.
  const Vector3<T>& p_MV = vertices_M_[elements_[e].vertex(i)];
  const Vector3<T>& p_MA = vertices_M_[elements_[e].vertex((i + 1) % 4)];
  const Vector3<T>& p_MB = vertices_M_[elements_[e].vertex((i + 2) % 4)];
  const Vector3<T>& p_MC = vertices_M_[elements_[e].vertex((i + 3) % 4)];

  const Vector3<T> p_AV_M = p_MV - p_MA;
  const Vector3<T> p_AB_M = p_MB - p_MA;
  const Vector3<T> p_AC_M = p_MC - p_MA;

  // Let bᵥ be the barycentric coordinate function corresponding to vertex V.
  // bᵥ is a linear function of the points in the tetrahedron.
  // bᵥ = 0 on the plane through triangle ABC.
  // bᵥ = 1 on the plane through V parallel to ABC.
  // Therefore, bᵥ changes fastest in the direction of the face normal vector
  // of ABC towards V. The rate of change is 1/h, where h is the
  // height of vertex V from the base ABC.
  //
  //    ──────────────V────────────── plane bᵥ = 1
  //                 ╱ ╲       ┊
  //                ╱   ╲      ┊         Triangle ABC is perpendicular to
  //               ╱     ╲     ┊ h       this view, so ABC looks like a line
  //              ╱       ╲    ┊         segment instead of a triangle.
  //             ╱    ↑∇bᵥ ╲   ┊
  //    ────────A━━━B━━━━━━━C──────── plane bᵥ = 0
  //
  // We conclude that ∇bᵥ is the vector of length 1/h that is perpendicular to
  // ABC and points into the tetrahedron.
  //
  // To calculate ∇bᵥ, consider the scalar triple product (AB x AC)⋅AV, which
  // is the signed volume of the parallelepiped spanned by AB, AC, and AV, and
  // consider the cross product AB x AC, which is the area vector of the
  // parallelogram spanned by AB and AC. We have:
  //
  //       ∇bᵥ = normal vector inversely proportional to height
  //           = area vector / signed volume
  //           = (AB x AC) / ((AB x AC)⋅AV)
  //
  // Consider a non-degenerate tetrahedron (tetrahedron with volume well above
  // zero) with vertices V₀,V₁,V₂,V₃ (note that vertex Vᵢ is the local
  // iᵗʰ vertex of the tetrahedron, *not* the global iᵗʰ vertex of the whole
  // mesh). Assume V₀,V₁,V₂,V₃ is in positive orientation, i.e.,
  // (V₁ - V₀)x(V₂ - V₀) points towards V₃. Given the vertex V = Vᵢ,
  // i ∈ {0,1,2,3}, the vertices A = Vᵢ₊₁, B = Vᵢ₊₂, C = Vᵢ₊₃ form the vector
  // AB x AC that points from ABC towards V when i is 1 or 3, and
  // AB x AC points away from V when i is 0 or 2. When AB x AC points towards
  // V, the signed volume (AB x AC)⋅AV is positive, and when AB x AC points
  // away from V, the signed volume is negative. As a result, the vector
  // ∇bᵥ = (AB x AC) / ((AB x AC)⋅AV) always points from ABC towards V as
  // expected.
  //
  // If the tetrahedron is degenerate (tetrahedron with almost zero volume),
  // the calculation (AB x AC) / ((AB x AC)⋅AV) is not numerically reliable any
  // more due to rounding errors. Near-zero-area triangle ABC may have the
  // area-vector calculation AB x AC pointing in the wrong direction. If ABC is
  // well formed but V is almost co-planar with ABC (V is near a vertex of ABC,
  // near the spanning line of an edge of ABC, or anywhere on the spanning plane
  // of ABC), the signed volume calculation ((AB x AC)⋅AV) may become a
  // near-zero number with the wrong sign. In these degenerate cases, we
  // throw std::exception.
  //
  const Vector3<T> area_vector_M = p_AB_M.cross(p_AC_M);  // AB x AC
  const T signed_volume = area_vector_M.dot(p_AV_M);      // (AB x AC)⋅AV

  constexpr double kEps = std::numeric_limits<double>::epsilon();

  // TODO(DamrongGuoy): Find a better way to handle degeneracy. Right now we
  //  check the volume of the parallelepiped against the threshold equal the
  //  machine epsilon. We might want to scale the threshold with the
  //  dimensions (length, area) of the parts of the tetrahedron (6 edges, 4
  //  triangles). Furthermore, we might also want case analysis for various
  //  kinds of degenerate tetrahedra; for example, a tetrahedron with
  //  one or more near-zero-length edges, a tetrahedron with one or more
  //  near-zero-area obtuse triangles without near-zero-length edges,
  //  or a tetrahedron with near-zero volume without near-zero-area triangles.

  using std::abs;
  if (abs(signed_volume) <= kEps) {
    return {};
  }
  return area_vector_M / signed_volume;
}

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class VolumeMesh);

}  // namespace geometry
}  // namespace drake
