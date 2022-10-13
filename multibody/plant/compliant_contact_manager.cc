#include "drake/multibody/plant/compliant_contact_manager.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/scope_exit.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/query_results/penetration_as_point_pair.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/sap_driver.h"
#include "drake/multibody/triangle_quadrature/gaussian_triangle_quadrature_rule.h"
#include "drake/systems/framework/context.h"

using drake::geometry::GeometryId;
using drake::geometry::PenetrationAsPointPair;
using drake::multibody::contact_solvers::internal::ContactSolverResults;
using drake::multibody::internal::DiscreteContactPair;
using drake::multibody::internal::MultibodyTreeTopology;
using drake::systems::Context;

namespace drake {
namespace multibody {
namespace internal {

template <typename T>
AccelerationsDueToExternalForcesCache<T>::AccelerationsDueToExternalForcesCache(
    const MultibodyTreeTopology& topology)
    : forces(topology.num_bodies(), topology.num_velocities()),
      abic(topology),
      Zb_Bo_W(topology.num_bodies()),
      aba_forces(topology),
      ac(topology) {}

template <typename T>
CompliantContactManager<T>::CompliantContactManager() = default;

template <typename T>
CompliantContactManager<T>::~CompliantContactManager() = default;

template <typename T>
void CompliantContactManager<T>::set_sap_solver_parameters(
    const contact_solvers::internal::SapSolverParameters& parameters) {
  DRAKE_DEMAND(sap_driver_ != nullptr);
  sap_driver_->set_sap_solver_parameters(parameters);
}

template <typename T>
void CompliantContactManager<T>::DeclareCacheEntries() {
  // N.B. We use xd_ticket() instead of q_ticket() since discrete
  // multibody plant does not have q's, but rather discrete state.
  // Therefore if we make it dependent on q_ticket() the Jacobian only
  // gets evaluated once at the start of the simulation.

  // Cache discrete contact pairs.
  const auto& discrete_contact_pairs_cache_entry = this->DeclareCacheEntry(
      "Discrete contact pairs.",
      systems::ValueProducer(
          this, &CompliantContactManager<T>::CalcDiscreteContactPairs),
      {systems::System<T>::xd_ticket(),
       systems::System<T>::all_parameters_ticket()});
  cache_indexes_.discrete_contact_pairs =
      discrete_contact_pairs_cache_entry.cache_index();

  // Accelerations due to non-contact forces.
  // We cache non-contact forces, ABA forces and accelerations into a
  // AccelerationsDueToExternalForcesCache.
  AccelerationsDueToExternalForcesCache<T> non_contact_forces_accelerations(
      this->internal_tree().get_topology());
  const auto& non_contact_forces_accelerations_cache_entry =
      this->DeclareCacheEntry(
          "Non-contact forces accelerations.",
          systems::ValueProducer(
              this, non_contact_forces_accelerations,
              &CompliantContactManager<
                  T>::CalcAccelerationsDueToNonContactForcesCache),
          // Due to issue #12786, we cannot properly mark this entry dependent
          // on inputs. CalcAccelerationsDueToNonContactForcesCache() uses
          // CacheIndexes::non_contact_forces_evaluation_in_progress to guard
          // against algebraic loops.
          {systems::System<T>::xd_ticket(),
           systems::System<T>::all_parameters_ticket()});
  cache_indexes_.non_contact_forces_accelerations =
      non_contact_forces_accelerations_cache_entry.cache_index();

  if constexpr (std::is_same_v<T, double>) {
    if (deformable_driver_ != nullptr) {
      deformable_driver_->DeclareCacheEntries(this);
    }
  }

  if (sap_driver_ != nullptr) sap_driver_->DeclareCacheEntries(this);
}

template <typename T>
std::vector<ContactPairKinematics<T>>
CompliantContactManager<T>::CalcContactKinematics(
    const systems::Context<T>& context) const {
  const std::vector<DiscreteContactPair<T>>& contact_pairs =
      this->EvalDiscreteContactPairs(context);
  const int num_contacts = contact_pairs.size();
  std::vector<ContactPairKinematics<T>> contact_kinematics;
  contact_kinematics.reserve(num_contacts);

  // Quick no-op exit.
  if (num_contacts == 0) return contact_kinematics;

  // Scratch workspace variables.
  const int nv = plant().num_velocities();
  Matrix3X<T> Jv_WAc_W(3, nv);
  Matrix3X<T> Jv_WBc_W(3, nv);
  Matrix3X<T> Jv_AcBc_W(3, nv);

  const Frame<T>& frame_W = plant().world_frame();
  for (int icontact = 0; icontact < num_contacts; ++icontact) {
    const auto& point_pair = contact_pairs[icontact];

    const GeometryId geometryA_id = point_pair.id_A;
    const GeometryId geometryB_id = point_pair.id_B;

    BodyIndex bodyA_index = this->geometry_id_to_body_index().at(geometryA_id);
    const Body<T>& bodyA = plant().get_body(bodyA_index);
    BodyIndex bodyB_index = this->geometry_id_to_body_index().at(geometryB_id);
    const Body<T>& bodyB = plant().get_body(bodyB_index);

    // Contact normal from point A into B.
    const Vector3<T>& nhat_W = -point_pair.nhat_BA_W;
    const Vector3<T>& p_WC = point_pair.p_WC;

    // Since v_AcBc_W = v_WBc - v_WAc the relative velocity Jacobian will be:
    //   J_AcBc_W = Jv_WBc_W - Jv_WAc_W.
    // That is the relative velocity at C is v_AcBc_W = J_AcBc_W * v.
    this->internal_tree().CalcJacobianTranslationalVelocity(
        context, JacobianWrtVariable::kV, bodyA.body_frame(), frame_W, p_WC,
        frame_W, frame_W, &Jv_WAc_W);
    this->internal_tree().CalcJacobianTranslationalVelocity(
        context, JacobianWrtVariable::kV, bodyB.body_frame(), frame_W, p_WC,
        frame_W, frame_W, &Jv_WBc_W);
    Jv_AcBc_W = Jv_WBc_W - Jv_WAc_W;

    // Define a contact frame C at the contact point such that the z-axis Cz
    // equals nhat_W. The tangent vectors are arbitrary, with the only
    // requirement being that they form a valid right handed basis with nhat_W.
    math::RotationMatrix<T> R_WC =
        math::RotationMatrix<T>::MakeFromOneVector(nhat_W, 2);

    const TreeIndex& treeA_index =
        tree_topology().body_to_tree_index(bodyA_index);
    const TreeIndex& treeB_index =
        tree_topology().body_to_tree_index(bodyB_index);
    // Sanity check, at least one must be valid.
    DRAKE_DEMAND(treeA_index.is_valid() || treeB_index.is_valid());

    // We have at most two blocks per contact.
    std::vector<typename ContactPairKinematics<T>::JacobianTreeBlock>
        jacobian_blocks;
    jacobian_blocks.reserve(2);

    // Tree A contribution to contact Jacobian Jv_W_AcBc_C.
    if (treeA_index.is_valid()) {
      Matrix3X<T> J = R_WC.matrix().transpose() *
                      Jv_AcBc_W.middleCols(
                          tree_topology().tree_velocities_start(treeA_index),
                          tree_topology().num_tree_velocities(treeA_index));
      jacobian_blocks.emplace_back(treeA_index, std::move(J));
    }

    // Tree B contribution to contact Jacobian Jv_W_AcBc_C.
    // This contribution must be added only if B is different from A.
    if ((treeB_index.is_valid() && !treeA_index.is_valid()) ||
        (treeB_index.is_valid() && treeB_index != treeA_index)) {
      Matrix3X<T> J = R_WC.matrix().transpose() *
                      Jv_AcBc_W.middleCols(
                          tree_topology().tree_velocities_start(treeB_index),
                          tree_topology().num_tree_velocities(treeB_index));
      jacobian_blocks.emplace_back(treeB_index, std::move(J));
    }

    contact_kinematics.emplace_back(point_pair.phi0, std::move(jacobian_blocks),
                                    std::move(R_WC));
  }

  return contact_kinematics;
}

template <typename T>
T CompliantContactManager<T>::GetPointContactStiffness(
    geometry::GeometryId id,
    const geometry::SceneGraphInspector<T>& inspector) const {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);
  // N.B. Here we rely on the resolution of #13289 and #5454 to get properties
  // with the proper scalar type T. This will not work on scalar converted
  // models until those issues are resolved.
  return prop->template GetPropertyOrDefault<T>(
      geometry::internal::kMaterialGroup, geometry::internal::kPointStiffness,
      this->default_contact_stiffness());
}

template <typename T>
T CompliantContactManager<T>::GetDissipationTimeConstant(
    geometry::GeometryId id,
    const geometry::SceneGraphInspector<T>& inspector) const {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);

  auto provide_context_string =
      [this, &inspector](geometry::GeometryId geometry_id) -> std::string {
    const BodyIndex body_index =
        this->geometry_id_to_body_index().at(geometry_id);
    const Body<T>& body = plant().get_body(body_index);
    return fmt::format("For geometry {} on body {}.",
                       inspector.GetName(geometry_id), body.name());
  };

  // N.B. Here we rely on the resolution of #13289 and #5454 to get properties
  // with the proper scalar type T. This will not work on scalar converted
  // models until those issues are resolved.
  const T relaxation_time = prop->template GetPropertyOrDefault<double>(
      geometry::internal::kMaterialGroup, "relaxation_time", 0.1);
  if (relaxation_time < 0.0) {
    const std::string message = fmt::format(
        "Relaxation time must be non-negative and relaxation_time "
        "= {} was provided. {}",
        relaxation_time, provide_context_string(id));
    throw std::runtime_error(message);
  }
  return relaxation_time;
}

template <typename T>
double CompliantContactManager<T>::GetCoulombFriction(
    geometry::GeometryId id,
    const geometry::SceneGraphInspector<T>& inspector) const {
  const geometry::ProximityProperties* prop =
      inspector.GetProximityProperties(id);
  DRAKE_DEMAND(prop != nullptr);
  DRAKE_THROW_UNLESS(prop->HasProperty(geometry::internal::kMaterialGroup,
                                       geometry::internal::kFriction));
  return prop
      ->GetProperty<CoulombFriction<double>>(geometry::internal::kMaterialGroup,
                                             geometry::internal::kFriction)
      .dynamic_friction();
}

template <typename T>
T CompliantContactManager<T>::CombineStiffnesses(const T& k1, const T& k2) {
  // Simple utility to detect 0 / 0. As it is used in this method, denom
  // can only be zero if num is also zero, so we'll simply return zero.
  auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? 0.0 : num / denom;
  };
  return safe_divide(k1 * k2, k1 + k2);
}

template <typename T>
T CompliantContactManager<T>::CombineDissipationTimeConstant(const T& tau1,
                                                             const T& tau2) {
  return tau1 + tau2;
}

template <typename T>
void CompliantContactManager<T>::CalcDiscreteContactPairs(
    const systems::Context<T>& context,
    std::vector<DiscreteContactPair<T>>* contact_pairs) const {
  plant().ValidateContext(context);
  DRAKE_DEMAND(contact_pairs != nullptr);

  contact_pairs->clear();
  if (plant().num_collision_geometries() == 0) return;

  const auto contact_model = plant().get_contact_model();

  // We first compute the number of contact pairs so that we can allocate all
  // memory at once.
  // N.B. num_point_pairs = 0 when:
  //   1. There are legitimately no point pairs or,
  //   2. the point pair model is not even in use.
  // We guard for case (2) since EvalPointPairPenetrations() cannot be called
  // when point contact is not used and would otherwise throw an exception.
  int num_point_pairs = 0;  // The number of point contact pairs.
  if (contact_model == ContactModel::kPoint ||
      contact_model == ContactModel::kHydroelasticWithFallback) {
    num_point_pairs = plant().EvalPointPairPenetrations(context).size();
  }

  int num_quadrature_pairs = 0;
  // N.B. For discrete hydro we use a first order quadrature rule. As such,
  // the per-face quadrature point is the face's centroid and the weight is 1.
  // This is compatible with a mesh that is triangle or polygon. If we attempted
  // higher order quadrature, polygons would have to be decomposed into smaller
  // n-gons which can receive an appropriate set of quadrature points.
  if (contact_model == ContactModel::kHydroelastic ||
      contact_model == ContactModel::kHydroelasticWithFallback) {
    const std::vector<geometry::ContactSurface<T>>& surfaces =
        this->EvalContactSurfaces(context);
    for (const auto& s : surfaces) {
      // One quadrature point per face.
      num_quadrature_pairs += s.num_faces();
    }
  }
  const int num_contact_pairs = num_point_pairs + num_quadrature_pairs;
  contact_pairs->reserve(num_contact_pairs);
  if (contact_model == ContactModel::kPoint ||
      contact_model == ContactModel::kHydroelasticWithFallback) {
    AppendDiscreteContactPairsForPointContact(context, contact_pairs);
  }
  if (contact_model == ContactModel::kHydroelastic ||
      contact_model == ContactModel::kHydroelasticWithFallback) {
    AppendDiscreteContactPairsForHydroelasticContact(context, contact_pairs);
  }
}

template <typename T>
void CompliantContactManager<T>::AppendDiscreteContactPairsForPointContact(
    const systems::Context<T>& context,
    std::vector<DiscreteContactPair<T>>* result) const {
  std::vector<DiscreteContactPair<T>>& contact_pairs = *result;

  const geometry::QueryObject<T>& query_object =
      this->plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const geometry::SceneGraphInspector<T>& inspector = query_object.inspector();

  // Simple utility to detect 0 / 0. As it is used in this method, denom
  // can only be zero if num is also zero, so we'll simply return zero.
  auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? T(0.0) : num / denom;
  };

  // Fill in the point contact pairs.
  const std::vector<PenetrationAsPointPair<T>>& point_pairs =
      plant().EvalPointPairPenetrations(context);
  for (const PenetrationAsPointPair<T>& pair : point_pairs) {
    const T kA = GetPointContactStiffness(pair.id_A, inspector);
    const T kB = GetPointContactStiffness(pair.id_B, inspector);
    const T k = CombineStiffnesses(kA, kB);
    const T tauA = GetDissipationTimeConstant(pair.id_A, inspector);
    const T tauB = GetDissipationTimeConstant(pair.id_B, inspector);
    const T tau = CombineDissipationTimeConstant(tauA, tauB);

    // Combine friction coefficients.
    const double muA = GetCoulombFriction(pair.id_A, inspector);
    const double muB = GetCoulombFriction(pair.id_B, inspector);
    const T mu = T(safe_divide(2.0 * muA * muB, muA + muB));

    // We compute the position of the point contact based on Hertz's theory
    // for contact between two elastic bodies.
    const T denom = kA + kB;
    const T wA = (denom == 0 ? 0.5 : kA / denom);
    const T wB = (denom == 0 ? 0.5 : kB / denom);
    const Vector3<T> p_WC = wA * pair.p_WCa + wB * pair.p_WCb;

    const T phi0 = -pair.depth;
    const T fn0 = NAN;  // not used.
    const T d = NAN;    // not used.
    contact_pairs.push_back({pair.id_A, pair.id_B, p_WC, pair.nhat_BA_W, phi0,
                             fn0, k, d, -1 /* invalide surface index */, tau,
                             mu});
  }
}

// Most of the calculation in this function should be the same as in
// MultibodyPlant<T>::CalcDiscreteContactPairs().
template <typename T>
void CompliantContactManager<T>::
    AppendDiscreteContactPairsForHydroelasticContact(
        const systems::Context<T>& context,
        std::vector<DiscreteContactPair<T>>* result) const {
  std::vector<DiscreteContactPair<T>>& contact_pairs = *result;

  // Simple utility to detect 0 / 0. As it is used in this method, denom
  // can only be zero if num is also zero, so we'll simply return zero.
  auto safe_divide = [](const T& num, const T& denom) {
    return denom == 0.0 ? 0.0 : num / denom;
  };

  // N.B. For discrete hydro we use a first order quadrature rule. As such,
  // the per-face quadrature point is the face's centroid and the weight is 1.
  // This is compatible with a mesh that is triangle or polygon. If we attempted
  // higher order quadrature, polygons would have to be decomposed into smaller
  // n-gons which can receive an appropriate set of quadrature points.

  const geometry::QueryObject<T>& query_object =
      this->plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const geometry::SceneGraphInspector<T>& inspector = query_object.inspector();
  const std::vector<geometry::ContactSurface<T>>& surfaces =
      this->EvalContactSurfaces(context);

  auto split_positive_area = [](const std::vector<Vector3<T>>& v,
                                const std::vector<T>& p,
                                std::vector<Vector3<T>>& new_v) {
    for (int i = 0; i < static_cast<int>(v.size()); ++i) {
      int j = (i + 1) % v.size();
      if (p[i] >= 0) {
        new_v.push_back(v[i]);
      }
      if ((p[i] > 0 && p[j] < 0) || (p[i] < 0 && p[j] > 0)) {
        const T t = p[i] / (p[i] - p[j]);
        new_v.push_back((1 - t) * v[i] + (t)*v[j]);
      }
    }
  };

  auto centroid_and_area = [](const std::vector<Vector3<T>>& v,
                              Vector3<T>& centroid, T& total_area) {
    for (int i = 1; i < static_cast<int>(v.size()) - 1; ++i) {
      const T area = (v[i] - v[0]).cross(v[i + 1] - v[0]).norm();
      total_area += area;
      centroid += area * (v[0] + v[i] + v[i+1]);
    }
    total_area /= 2;
    centroid /= 6;
    if (total_area > 0) centroid /= total_area;
  };

  const int num_surfaces = surfaces.size();
  for (int surface_index = 0; surface_index < num_surfaces; ++surface_index) {

    // std::cout << "\n\n\n\n\nsurface: " << surface_index << "\n";

    const auto& s = surfaces[surface_index];
    const bool M_is_compliant = s.HasGradE_M();
    const bool N_is_compliant = s.HasGradE_N();
    DRAKE_DEMAND(M_is_compliant || N_is_compliant);

    BodyIndex bodyA_index = this->geometry_id_to_body_index().at(s.id_M());
    const Body<T>& bodyA = plant().get_body(bodyA_index);
    BodyIndex bodyB_index = this->geometry_id_to_body_index().at(s.id_N());
    const Body<T>& bodyB = plant().get_body(bodyB_index);

    const SpatialVelocity<T>& V_WA = bodyA.EvalSpatialVelocityInWorld(context);
    const SpatialVelocity<T>& V_WB = bodyB.EvalSpatialVelocityInWorld(context);

    // Combine dissipation.
    const T tau_M = GetDissipationTimeConstant(s.id_M(), inspector);
    const T tau_N = GetDissipationTimeConstant(s.id_N(), inspector);
    const T tau = CombineDissipationTimeConstant(tau_M, tau_N);

    // Combine friction coefficients.
    const double muA = GetCoulombFriction(s.id_M(), inspector);
    const double muB = GetCoulombFriction(s.id_N(), inspector);
    const T mu = T(safe_divide(2.0 * muA * muB, muA + muB));

    int reject = 0;

    auto extract_verts_and_pressure = [&s](int face, std::vector<Vector3<T>>& v,
                                           std::vector<T>& p) {
      int num_v = s.is_triangle()
                      ? s.tri_mesh_W().element(face).num_vertices()
                      : s.poly_mesh_W().element(face).num_vertices();
      for (int i = 0; i < num_v; ++i) {
        v.push_back(
            s.is_triangle()
                ? s.tri_mesh_W().vertex(s.tri_mesh_W().element(face).vertex(i))
                : s.poly_mesh_W().vertex(
                      s.poly_mesh_W().element(face).vertex(i)));
        p.push_back(s.is_triangle()
                        ? s.tri_e_MN().EvaluateAtVertex(
                              s.tri_mesh_W().element(face).vertex(i))
                        : s.poly_e_MN().EvaluateAtVertex(
                              s.poly_mesh_W().element(face).vertex(i)));
      }
    };

    for (int face = 0; face < s.num_faces(); ++face) {
      Vector3<T> p_WQ(0, 0, 0);
      T Ae(0);

      // Partially negative face rectification
      std::vector<Vector3<T>> v;
      std::vector<T> p;
      extract_verts_and_pressure(face, v, p);

      // Count the positive and negative pressure verts
      int num_pos = 0;
      int num_neg = 0;
      for (const T& val : p) {
        if (val > 0) ++num_pos;
        if (val < 0) ++num_neg;
      }

      // If this is a mixed face, rectify. Otherwise compute the area and
      // centroid as normal.
      if (num_pos > 0 && num_neg > 0) {
        std::vector<Vector3<T>> new_v;
        split_positive_area(v, p, new_v);
        centroid_and_area(new_v, p_WQ, Ae);
      } else {
        Ae = s.area(face);  // Face element area.
        p_WQ = s.centroid(face);
      }

      // if (surface_index == 1 && Ae != s.area(face)) {
      //   std::cout << fmt::format("v size: {} new v size: {}\n", v.size(),
      //                            new_v.size());
      //   std::cout << "p: [";
      //   for(auto x : p) std::cout << x << ", ";
      //   std::cout <<"]\n";

      //   if (v.size() == new_v.size()) {
      //     for (int i = 0; i < static_cast<int>(v.size()); ++i) {
      //       std::cout << "    v: " << v[i].x() << "\t" << v[i].y() << "\t"
      //                 << v[i].z() << "\n";
      //       std::cout << "new_v: " << new_v[i].x() << "\t" << new_v[i].y()
      //                 << "\t" << new_v[i].z() << "\n\n";
      //     }
      //   }

      //   std::cout << "    p_WQ: " << p_WQ.x() << "\t" << p_WQ.y() << "\t"
      //             << p_WQ.z() << "\n";
      //   std::cout << "centroid: " << s.centroid(face).x() << "\t"
      //             << s.centroid(face).y() << "\t" << s.centroid(face).z()
      //             << "\n\n";
      // }

      // std::cout << fmt::format("Ae: {} Area: {}\n", Ae, s.area(face));

      // const T& Ae = s.area(face);  // Face element area.

      // We found out that the hydroelastic query might report
      // infinitesimally small triangles (consider for instance an initial
      // condition that perfectly places an object at zero distance from the
      // ground.) While the area of zero sized triangles is not a problem by
      // itself, the badly computed normal on these triangles leads to
      // problems when computing the contact Jacobians (since we need to
      // obtain an orthonormal basis based on that normal.)
      // We therefore ignore infinitesimally small triangles. The tolerance
      // below is somehow arbitrary and could possibly be tightened.
      if (Ae > 1.0e-14) {
        // From ContactSurface's documentation: The normal of each face is
        // guaranteed to point "out of" N and "into" M.
        const Vector3<T>& nhat_W = s.face_normal(face);

        // One dimensional pressure gradient (in Pa/m). Unlike [Masterjohn
        // 2022], for convenience we define both pressure gradients
        // to be positive in the direction "into" the bodies. Therefore,
        // we use the minus sign for gN.
        // [Masterjohn 2022] Velocity Level Approximation of Pressure
        // Field Contact Patches.
        const T gM = M_is_compliant
                         ? s.EvaluateGradE_M_W(face).dot(nhat_W)
                         : T(std::numeric_limits<double>::infinity());
        const T gN = N_is_compliant
                         ? -s.EvaluateGradE_N_W(face).dot(nhat_W)
                         : T(std::numeric_limits<double>::infinity());

        constexpr double kGradientEpsilon = 1.0e-14;
        if (gM < kGradientEpsilon || gN < kGradientEpsilon) {
          // Mathematically g = gN*gM/(gN+gM) and therefore g = 0 when
          // either gradient on one of the bodies is zero. A zero gradient
          // means there is no contact constraint, and therefore we
          // ignore it to avoid numerical problems in the discrete solver.
          continue;
        }

        // Effective hydroelastic pressure gradient g result of
        // compliant-compliant interaction, see [Masterjohn 2022].
        // The expression below is mathematically equivalent to g =
        // gN*gM/(gN+gM) but it has the advantage of also being valid if
        // one of the gradients is infinity.
        const T g = 1.0 / (1.0 / gM + 1.0 / gN);

        // Position of quadrature point Q in the world frame (since mesh_W
        // is measured and expressed in W).

        // const Vector3<T>& p_WQ = s.centroid(face);

        // For a triangle, its centroid has the fixed barycentric
        // coordinates independent of the shape of the triangle. Using
        // barycentric coordinates to evaluate field value could be
        // faster than using Cartesian coordiantes, especially if the
        // TriangleSurfaceMeshFieldLinear<> does not store gradients and
        // has to solve linear equations to convert Cartesian to
        // barycentric coordinates.

        // const Vector3<T> tri_centroid_barycentric(1 / 3., 1 / 3., 1 / 3.);

        // Pressure at the quadrature point.

        // const T p0 = s.is_triangle()
        //                  ? s.tri_e_MN().Evaluate(face,
        //                  tri_centroid_barycentric) :
        //                  s.poly_e_MN().EvaluateCartesian(face, p_WQ);
        const T p0 = s.is_triangle()
                         ? s.tri_e_MN().EvaluateCartesian(face, p_WQ)
                         : s.poly_e_MN().EvaluateCartesian(face, p_WQ);

        // std::cout << fmt::format(
        //     "p0: {} pressure: {}\n", p0,
        //     s.is_triangle()
        //         ? s.tri_e_MN().Evaluate(face, tri_centroid_barycentric)
        //         : s.poly_e_MN().EvaluateCartesian(face, s.centroid(face)));
        // std::cout << "-------------------------------------\n";

        // Effective compliance in the normal direction for the given
        // discrete patch, refer to [Masterjohn 2022] for details.
        // [Masterjohn 2022] Masterjohn J., Guoy D., Shepherd J. and
        // Castro A., 2022. Velocity Level Approximation of Pressure Field
        // Contact Patches. Available at https://arxiv.org/abs/2110.04157.
        const T k = Ae * g;

        // phi < 0 when in penetration.
        const T phi0 = -p0 / g;

        const SpatialVelocity<T>& V_WAq = V_WA.Shift(p_WQ);
        const SpatialVelocity<T>& V_WBq = V_WB.Shift(p_WQ);
        const Vector3<T>& v_AqBq_W =
            V_WBq.translational() - V_WAq.translational();
        const T& v_n = -v_AqBq_W.dot(nhat_W);

        if (p0 < 0 && (p0 / (v_n * g)) > plant().time_step()) {
          ++reject;
          //continue;
        }

        if (k > 0) {
          const T fn0 = NAN;  // not used.
          const T d = NAN;    // not used.
          contact_pairs.push_back({s.id_M(), s.id_N(), p_WQ, nhat_W, phi0, fn0,
                                   k, d, surface_index, tau, mu});
        }
      }
    }

    // std::cout << fmt::format("total: {} reject: {}\n", s.num_faces(),
    // reject);
  }
}

template <typename T>
void CompliantContactManager<T>::CalcNonContactForcesExcludingJointLimits(
    const systems::Context<T>& context, MultibodyForces<T>* forces) const {
  DRAKE_DEMAND(forces != nullptr);
  DRAKE_DEMAND(forces->CheckHasRightSizeForModel(plant()));
  // Compute forces applied through force elements. Note that this resets
  // forces to empty so must come first.
  this->CalcForceElementsContribution(context, forces);
  this->AddInForcesFromInputPorts(context, forces);
}

template <typename T>
void CompliantContactManager<T>::CalcAccelerationsDueToNonContactForcesCache(
    const systems::Context<T>& context,
    AccelerationsDueToExternalForcesCache<T>* forward_dynamics_cache) const {
  DRAKE_DEMAND(forward_dynamics_cache != nullptr);
  ScopeExit guard = this->ThrowIfNonContactForceInProgress(context);

  // N.B. Joint limits are modeled as constraints. Therefore here we only add
  // all other external forces.
  CalcNonContactForcesExcludingJointLimits(context,
                                           &forward_dynamics_cache->forces);

  // Our goal is to compute accelerations from the Newton-Euler equations:
  //   M⋅v̇ = k(x)
  // where k(x) includes continuous forces of the state x not from constraints
  // such as force elements, Coriolis terms, actuation through input ports and
  // joint damping. We use a discrete time stepping scheme with time step dt
  // and accelerations
  //   v̇ = (v-v₀)/dt
  // where v₀ are the previous time step generalized velocities. We split
  // generalized forces k(x) as:
  //   k(x) = k₁(x) - D⋅v
  // where k₁(x) includes all other force contributions except damping and D
  // is the non-negative diagonal matrix for damping. Using this split, we
  // evaluate dissipation "implicitly" using the next time step velocities and
  // every other force in k₁(x) "explicitly" at the previous time step state
  // x₀. In total, our discrete update for the free motion velocities reads:
  //   M⋅(v-v₀)/dt = k₁(x₀) - D⋅v
  // We can rewrite this by adding and subtracting -D⋅v₀ on the right hand
  // side:
  //   M⋅(v-v₀)/dt = k₁(x₀) - D⋅(v-v₀) - D⋅v₀
  // which can be rearranged as:
  //   (M + dt⋅D)⋅(v-v₀)/dt = k₁(x₀) - D⋅v₀ = k(x₀)
  // Therefore the generalized accelerations a = (v-v₀)/dt can be computed
  // using ABA forward dynamics with non-constraint continuous forces
  // evaluated at x₀ and the addition of the diagonal term dt⋅D. We do this
  // below in terms of MultibodyTree APIs.

  // We must include reflected rotor inertias along with the new term dt⋅D.
  const VectorX<T> diagonal_inertia =
      plant().EvalReflectedInertiaCache(context) +
      joint_damping_ * plant().time_step();

  // We compute the articulated body inertia including the contribution of the
  // additional diagonal elements arising from the implicit treatment of joint
  // damping.
  this->internal_tree().CalcArticulatedBodyInertiaCache(
      context, diagonal_inertia, &forward_dynamics_cache->abic);
  this->internal_tree().CalcArticulatedBodyForceBias(
      context, forward_dynamics_cache->abic, &forward_dynamics_cache->Zb_Bo_W);
  this->internal_tree().CalcArticulatedBodyForceCache(
      context, forward_dynamics_cache->abic, forward_dynamics_cache->Zb_Bo_W,
      forward_dynamics_cache->forces, &forward_dynamics_cache->aba_forces);
  this->internal_tree().CalcArticulatedBodyAccelerations(
      context, forward_dynamics_cache->abic, forward_dynamics_cache->aba_forces,
      &forward_dynamics_cache->ac);
}

template <typename T>
const std::vector<DiscreteContactPair<T>>&
CompliantContactManager<T>::EvalDiscreteContactPairs(
    const systems::Context<T>& context) const {
  return plant()
      .get_cache_entry(cache_indexes_.discrete_contact_pairs)
      .template Eval<std::vector<DiscreteContactPair<T>>>(context);
}

template <typename T>
const multibody::internal::AccelerationKinematicsCache<T>&
CompliantContactManager<T>::EvalAccelerationsDueToNonContactForcesCache(
    const systems::Context<T>& context) const {
  return plant()
      .get_cache_entry(cache_indexes_.non_contact_forces_accelerations)
      .template Eval<AccelerationsDueToExternalForcesCache<T>>(context)
      .ac;
}

template <typename T>
void CompliantContactManager<T>::DoCalcContactSolverResults(
    const systems::Context<T>& context,
    ContactSolverResults<T>* contact_results) const {
  // TODO(amcastro-tri): Remove this DRAKE_DEMAND when other solvers are
  // supported.
  DRAKE_DEMAND(plant().get_discrete_contact_solver() ==
                   DiscreteContactSolver::kSap &&
               sap_driver_ != nullptr);
  sap_driver_->CalcContactSolverResults(context, contact_results);
}

template <typename T>
void CompliantContactManager<T>::DoCalcDiscreteValues(
    const drake::systems::Context<T>& context,
    drake::systems::DiscreteValues<T>* updates) const {
  const ContactSolverResults<T>& results =
      this->EvalContactSolverResults(context);

  // Previous time step positions.
  const int nq = plant().num_positions();
  const VectorX<T>& x0 =
      context.get_discrete_state(this->multibody_state_index()).value();
  const auto q0 = x0.topRows(nq);

  // Retrieve the solution velocity for the next time step.
  const VectorX<T>& v_next = results.v_next;

  // Update generalized positions.
  VectorX<T> qdot_next(plant().num_positions());
  plant().MapVelocityToQDot(context, v_next, &qdot_next);
  const VectorX<T> q_next = q0 + plant().time_step() * qdot_next;

  VectorX<T> x_next(plant().num_multibody_states());
  x_next << q_next, v_next;
  updates->set_value(this->multibody_state_index(), x_next);
}

// TODO(xuchenhan-tri): Consider a scalar converting constructor to cut down
// repeated code in CloneToDouble() and CloneToAutoDiffXd().
template <typename T>
std::unique_ptr<DiscreteUpdateManager<double>>
CompliantContactManager<T>::CloneToDouble() const {
  // Create a manager with default SAP parameters.
  auto clone = std::make_unique<CompliantContactManager<double>>();
  // N.B. we should copy/clone all members except for those overwritten in
  // ExtractModelInfo and DeclareCacheEntries.
  // E.g. SapParameters for SapDriver won't be the same after the clone.
  return clone;
}

template <typename T>
std::unique_ptr<DiscreteUpdateManager<AutoDiffXd>>
CompliantContactManager<T>::CloneToAutoDiffXd() const {
  // Create a manager with default SAP parameters.
  auto clone = std::make_unique<CompliantContactManager<AutoDiffXd>>();
  // N.B. we should copy/clone all members except for those overwritten in
  // ExtractModelInfo and DeclareCacheEntries.
  // E.g. SapParameters for SapDriver won't be the same after the clone.
  return clone;
}

template <typename T>
void CompliantContactManager<T>::ExtractModelInfo() {
  // Collect joint damping coefficients into a vector.
  joint_damping_ = VectorX<T>::Zero(plant().num_velocities());
  for (JointIndex j(0); j < plant().num_joints(); ++j) {
    const Joint<T>& joint = plant().get_joint(j);
    const int velocity_start = joint.velocity_start();
    const int nv = joint.num_velocities();
    joint_damping_.segment(velocity_start, nv) = joint.damping_vector();
  }

  // TODO(amcastro-tri): Remove this DRAKE_DEMAND when other solvers are
  // supported.
  DRAKE_DEMAND(plant().get_discrete_contact_solver() ==
                   DiscreteContactSolver::kSap &&
               sap_driver_ == nullptr);
  sap_driver_ = std::make_unique<SapDriver<T>>(this);

  // Collect information from each PhysicalModel owned by the plant.
  const std::vector<std::unique_ptr<multibody::internal::PhysicalModel<T>>>&
      physical_models = this->plant().physical_models();
  for (const auto& model : physical_models) {
    std::visit(
        [this](auto&& concrete_model) {
          this->ExtractConcreteModel(concrete_model);
        },
        model->ToPhysicalModelPointerVariant());
  }
}

template <typename T>
void CompliantContactManager<T>::ExtractConcreteModel(
    const DeformableModel<T>* model) {
  if constexpr (std::is_same_v<T, double>) {
    DRAKE_DEMAND(model != nullptr);
    // TODO(xuchenhan-tri): Demote this to a DRAKE_DEMAND when we check for
    //  duplicated model with MbP::AddPhysicalModel.
    if (deformable_driver_ != nullptr) {
      throw std::logic_error(
          fmt::format("{}: A deformable model has already been registered. "
                      "Repeated registration is not allowed.",
                      __func__));
    }
    deformable_driver_ =
        std::make_unique<DeformableDriver<double>>(model, this);
  } else {
    unused(model);
    throw std::logic_error(
        "Only T = double is supported for the simulation of deformable "
        "bodies.");
  }
}

template <typename T>
void CompliantContactManager<T>::DoCalcAccelerationKinematicsCache(
    const systems::Context<T>& context0,
    multibody::internal::AccelerationKinematicsCache<T>* ac) const {
  // Current state.
  const VectorX<T>& x0 =
      context0.get_discrete_state(this->multibody_state_index()).value();
  const auto v0 = x0.bottomRows(plant().num_velocities());

  // Next state.
  const ContactSolverResults<T>& results =
      this->EvalContactSolverResults(context0);
  const VectorX<T>& v_next = results.v_next;

  ac->get_mutable_vdot() = (v_next - v0) / plant().time_step();

  this->internal_tree().CalcSpatialAccelerationsFromVdot(
      context0, plant().EvalPositionKinematics(context0),
      plant().EvalVelocityKinematics(context0), ac->get_vdot(),
      &ac->get_mutable_A_WB_pool());
}

template <typename T>
void CompliantContactManager<T>::DoCalcHydroelasticContactForcesDiscrete(
    const systems::Context<T>& context,
    internal::HydroelasticContactInfoAndBodySpatialForces<T>*
        contact_info_and_body_forces) const {
  DRAKE_DEMAND(contact_info_and_body_forces != nullptr);

  std::vector<SpatialForce<T>>& F_BBo_W_array =
      contact_info_and_body_forces->F_BBo_W_array;
  DRAKE_DEMAND(static_cast<int>(F_BBo_W_array.size()) == plant().num_bodies());
  std::vector<HydroelasticContactInfo<T>>& contact_info =
      contact_info_and_body_forces->contact_info;
  if (plant().num_collision_geometries() == 0) return;

  const std::vector<drake::geometry::ContactSurface<T>>& all_surfaces =
      this->EvalContactSurfaces(context);

  // This method expect that the continuous version of this methid,
  // CalcHydroelasticContactForces(), has already been called. Therefore info
  // and forces have previously been allocated.
  DRAKE_DEMAND(static_cast<int>(F_BBo_W_array.size()) == plant().num_bodies());
  DRAKE_DEMAND(contact_info.size() == all_surfaces.size());

  // const auto& query_object = EvalGeometryQueryInput(context);
  // const geometry::SceneGraphInspector<T>& inspector =
  // query_object.inspector();

  const std::vector<internal::DiscreteContactPair<T>>& discrete_pairs =
      this->EvalDiscreteContactPairs(context);
  const std::vector<math::RotationMatrix<T>>& R_WC_set =
      this->sap_driver_->EvalContactProblemCache(context).R_WC;
  const contact_solvers::internal::ContactSolverResults<T>& solver_results =
      this->EvalContactSolverResults(context);

  const VectorX<T>& fn = solver_results.fn;
  const VectorX<T>& ft = solver_results.ft;
  const VectorX<T>& vt = solver_results.vt;
  const VectorX<T>& vn = solver_results.vn;

  // Discrete pairs contain both point and hydro contact force results.
  const int num_contacts = discrete_pairs.size();
  DRAKE_DEMAND(fn.size() == num_contacts);
  DRAKE_DEMAND(ft.size() == 2 * num_contacts);
  DRAKE_DEMAND(vn.size() == num_contacts);
  DRAKE_DEMAND(vt.size() == 2 * num_contacts);

  int num_point_contacts = 0;
  for (auto pair : discrete_pairs) {
    if (pair.surface_index == -1) ++num_point_contacts;
  }
  const int num_surfaces = all_surfaces.size();

  std::vector<SpatialForce<T>> contact_surface_forces(num_surfaces,
                                                      SpatialForce<T>::Zero());

  // We only scan discrete pairs corresponding to hydroelastic quadrature
  // points. These are appended at the end of the point contact forces.
  for (int icontact = num_point_contacts; icontact < num_contacts; ++icontact) {
    const auto& pair = discrete_pairs[icontact];
    // const GeometryId geometryA_id = pair.id_A;
    // const GeometryId geometryB_id = pair.id_B;
    // const BodyIndex bodyA_index =
    // geometry_id_to_body_index_.at(geometryA_id); const BodyIndex bodyB_index
    // = geometry_id_to_body_index_.at(geometryB_id);

    // Quadrature point Q.
    const Vector3<T>& p_WQ = pair.p_WC;
    const math::RotationMatrix<T>& R_WC = R_WC_set[icontact];

    // Contact forces applied on B at quadrature point Q.
    const Vector3<T> f_Bq_C(ft(2 * icontact), ft(2 * icontact + 1),
                            -fn(icontact));
    const Vector3<T> f_Bq_W = R_WC * f_Bq_C;

    const auto& s = all_surfaces[pair.surface_index];
    // Surface's centroid point O.
    const Vector3<T>& p_WO = s.is_triangle() ? s.tri_mesh_W().centroid()
                                             : s.poly_mesh_W().centroid();

    // Torque about the centroid.
    const Vector3<T> p_OQ_W = p_WQ - p_WO;
    const Vector3<T> t_Bo_W = p_OQ_W.cross(f_Bq_W);

    // Accumulate force for the corresponding contact surface.
    contact_surface_forces[pair.surface_index] +=
        SpatialForce<T>(t_Bo_W, f_Bq_W);
  }

  // Update contact info to include the correct contact forces.
  for (int surface_index = 0; surface_index < num_surfaces; ++surface_index) {
    auto& info = contact_info[surface_index];
    info.mutable_F_Ac_W() = contact_surface_forces[surface_index];
  }
}

}  // namespace internal
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::internal::CompliantContactManager);
