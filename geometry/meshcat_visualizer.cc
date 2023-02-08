#include "drake/geometry/meshcat_visualizer.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <array>
#include <set>

#include <fmt/format.h>

#include "drake/common/extract_double.h"
#include "drake/geometry/utilities.h"
#include "drake/geometry/proximity/sorted_triplet.h"

namespace drake {
namespace geometry {

namespace {

/* Analyzes the tetrahedral mesh topology of the deformble geometry with `g_id`
 to do the following:
  1. Build a surface mesh from each volume mesh.
  2. Create a mapping from surface vertex to volume vertex for each mesh.
  3. Record the expected number of vertices referenced by each tet mesh.
 Store the results in MeshcatDeformableMeshData.
 @pre g_id corresponds to a deformable geometry. */
template <typename T>
internal::MeshcatDeformableMeshData MakeMeshcatDeformableMeshData(
    GeometryId g_id, const SceneGraphInspector<T>& inspector) {
  /* For each tet mesh, extract all the border triangles. Those are the
   triangles that are only referenced by a single tet. So, for every tet, we
   examine its four constituent triangle and determine if any other tet
   shares it. Any triangle that is only referenced once is a border triangle.
   Each triangle has a unique key: a SortedTriplet (so the ordering of the
   triangle vertex indices won't matter). The first time we see a triangle, we
   add it to a map. The second time we see the triangle, we remove it. When
   we're done, the keys in the map will be those triangles referenced only once.
   The values in the map represent the triangle, with the vertex indices
   ordered so that they point *out* of the tetrahedron. Therefore,
   they will also point outside of the mesh. A typical tetrahedral element
   looks like:

       p2 *
          |
          |
       p3 *---* p0
         /
        /
    p1 *

   The index order for a particular tetrahedron has the order [p0, p1, p2,
   p3]. These local indices enumerate each of the tet triangles with
   outward-pointing normals with respect to the right-hand rule. */
  const std::array<std::array<int, 3>, 4> local_indices{
      {{{1, 0, 2}}, {{3, 0, 1}}, {{3, 1, 2}}, {{2, 0, 3}}}};

  const VolumeMesh<double>* mesh = inspector.GetReferenceMesh(g_id);
  DRAKE_DEMAND(mesh != nullptr);

  std::map<internal::SortedTriplet<int>, std::array<int, 3>> border_triangles;
  for (const VolumeElement& tet : mesh->tetrahedra()) {
    for (const std::array<int, 3>& tet_triangle : local_indices) {
      const std::array<int, 3> tri{tet.vertex(tet_triangle[0]),
                                   tet.vertex(tet_triangle[1]),
                                   tet.vertex(tet_triangle[2])};
      const internal::SortedTriplet triangle_key(tri[0], tri[1], tri[2]);
      // Here we rely on the fact that at most two tets would share a common
      // triangle.
      if (auto itr = border_triangles.find(triangle_key);
          itr != border_triangles.end()) {
        border_triangles.erase(itr);
      } else {
        border_triangles[triangle_key] = tri;
      }
    }
  }
  /* Record the expected minimum number of vertex positions to be received.
   For simplicity we choose a generous upper bound: the total number of
   vertices in the tetrahedral mesh, even though we really only need the
   positions of the vertices on the surface. */
  // TODO(xuchenhan-tri) It might be worthwhile to make largest_index the
  //  largest index that lies on the surface. Then, when we create our meshes,
  //  if we intentionally construct them so that the surface vertices come
  //  first, we will process a very compact representation.
  const int volume_vertex_count = mesh->num_vertices();

  /* Using a set because the vertices will be nicely ordered. Ideally, we'll
   be extracting a subset of the vertex positions from the input port. We
   optimize cache coherency if we march in a monotonically increasing pattern.
   So, we'll map triangle vertex indices to volume vertex indices in a
   strictly monotonically increasing relationship. */
  std::set<int> unique_vertices;
  for (const auto& [triangle_key, triangle] : border_triangles) {
    unused(triangle_key);
    for (int j = 0; j < 3; ++j) unique_vertices.insert(triangle[j]);
  }

  /* This is the *second* documented responsibility of this function: Populate
   the mapping from surface to volume so that we can efficiently extract the
   *surface* vertex positions from the *volume* vertex input. */
  std::vector<int> surface_to_volume_vertices;
  surface_to_volume_vertices.insert(surface_to_volume_vertices.begin(),
                                    unique_vertices.begin(),
                                    unique_vertices.end());

  /* The border triangles all include indices into the volume vertices. To turn
   them into surface triangles, they need to include indices into the surface
   vertices. Create the volume index --> surface map to facilitate the
   transformation. */
  const int surface_vertex_count =
      static_cast<int>(surface_to_volume_vertices.size());
  std::map<int, int> volume_to_surface;
  for (int j = 0; j < surface_vertex_count; ++j) {
    volume_to_surface[surface_to_volume_vertices[j]] = j;
  }

  /* This is the *first* documented responsibility: Create the topology of the
   surface triangle mesh for each volume mesh. Each triangle consists of three
   indices into the set of *surface* vertex positions. */
  std::vector<Vector3<int>> surface_triangles;
  surface_triangles.reserve(border_triangles.size());
  for (auto& [triangle_key, face] : border_triangles) {
    unused(triangle_key);
    surface_triangles.emplace_back(volume_to_surface[face[0]],
                                   volume_to_surface[face[1]],
                                   volume_to_surface[face[2]]);
  }

  // TODO(xuchenhan-tri): Read the color of the mesh from properties.
  return {g_id,
          inspector.GetName(g_id),
          move(surface_to_volume_vertices),
          move(surface_triangles),
          volume_vertex_count};
}

}  // namespace

template <typename T>
MeshcatVisualizer<T>::MeshcatVisualizer(std::shared_ptr<Meshcat> meshcat,
                                        MeshcatVisualizerParams params)
    : systems::LeafSystem<T>(systems::SystemTypeTag<MeshcatVisualizer>{}),
      meshcat_(std::move(meshcat)),
      params_(std::move(params)),
      animation_(
          std::make_unique<MeshcatAnimation>(1.0 / params_.publish_period)),
      alpha_slider_name_(std::string(params_.prefix + " α")) {
  DRAKE_DEMAND(meshcat_ != nullptr);
  DRAKE_DEMAND(params_.publish_period >= 0.0);
  if (params_.role == Role::kUnassigned) {
    throw std::runtime_error(
        "MeshcatVisualizer cannot be used for geometries with the "
        "Role::kUnassigned value. Please choose kProximity, kPerception, or "
        "kIllustration");
  }

  this->DeclarePeriodicPublishEvent(params_.publish_period, 0.0,
                                    &MeshcatVisualizer<T>::UpdateMeshcat);
  this->DeclareForcedPublishEvent(&MeshcatVisualizer<T>::UpdateMeshcat);

  if (params_.delete_on_initialization_event) {
    this->DeclareInitializationPublishEvent(
        &MeshcatVisualizer<T>::OnInitialization);
  }

  query_object_input_port_ =
      this->DeclareAbstractInputPort("query_object", Value<QueryObject<T>>())
          .get_index();

  if (params_.enable_alpha_slider) {
    meshcat_->AddSlider(
      alpha_slider_name_, 0.02, 1.0, 0.02, alpha_value_);
  }

  // This cache entry depends on nothing. It will be marked as dirty when the
  // geometry version changes.
  deformable_data_cache_index_ =
      this->DeclareCacheEntry("deformable_data",
                              &MeshcatVisualizer<T>::CalcMeshcatDeformableMeshData,
                              {this->nothing_ticket()})
          .cache_index();
}

template <typename T>
template <typename U>
MeshcatVisualizer<T>::MeshcatVisualizer(const MeshcatVisualizer<U>& other)
    : MeshcatVisualizer(other.meshcat_, other.params_) {}

template <typename T>
void MeshcatVisualizer<T>::Delete() const {
  meshcat_->Delete(params_.prefix);
  version_ = GeometryVersion();
}

template <typename T>
void MeshcatVisualizer<T>::PublishRecording() const {
  meshcat_->SetAnimation(*animation_);
}

template <typename T>
void MeshcatVisualizer<T>::DeleteRecording() {
  animation_ = std::make_unique<MeshcatAnimation>(1.0 / params_.publish_period);
}

template <typename T>
MeshcatVisualizer<T>& MeshcatVisualizer<T>::AddToBuilder(
    systems::DiagramBuilder<T>* builder, const SceneGraph<T>& scene_graph,
    std::shared_ptr<Meshcat> meshcat, MeshcatVisualizerParams params) {
  return AddToBuilder(builder, scene_graph.get_query_output_port(),
                      std::move(meshcat), std::move(params));
}

template <typename T>
MeshcatVisualizer<T>& MeshcatVisualizer<T>::AddToBuilder(
    systems::DiagramBuilder<T>* builder,
    const systems::OutputPort<T>& query_object_port,
    std::shared_ptr<Meshcat> meshcat, MeshcatVisualizerParams params) {
  auto& visualizer = *builder->template AddSystem<MeshcatVisualizer<T>>(
      std::move(meshcat), std::move(params));
  builder->Connect(query_object_port, visualizer.query_object_input_port());
  return visualizer;
}

template <typename T>
systems::EventStatus MeshcatVisualizer<T>::UpdateMeshcat(
    const systems::Context<T>& context) const {
  const auto& query_object =
      query_object_input_port().template Eval<QueryObject<T>>(context);
  const GeometryVersion& current_version =
      query_object.inspector().geometry_version();

  if (!version_.IsSameAs(current_version, params_.role)) {
    SetObjects(query_object.inspector());
    RefreshMeshcatDeformableMeshData(context);
    version_ = current_version;
  }
  SetTransforms(context, query_object);
  SetDeformableMeshes(context, query_object, EvalMeshcatDeformableMeshData(context));
  if (params_.enable_alpha_slider) {
    double new_alpha_value = meshcat_->GetSliderValue(alpha_slider_name_);
    if (new_alpha_value != alpha_value_) {
      alpha_value_ = new_alpha_value;
      SetColorAlphas();
    }
  }
  std::optional<double> rate = realtime_rate_calculator_.UpdateAndRecalculate(
      ExtractDoubleOrThrow(context.get_time()));
  if (rate) {
    meshcat_->SetRealtimeRate(rate.value());
  }

  return systems::EventStatus::Succeeded();
}

template <typename T>
void MeshcatVisualizer<T>::SetObjects(
    const SceneGraphInspector<T>& inspector) const {
  colors_.clear();

  // Frames registered previously that are not set again here should be deleted.
  std::map <FrameId, std::string> frames_to_delete{};
  dynamic_frames_.swap(frames_to_delete);

  // Geometries registered previously that are not set again here should be
  // deleted.
  std::map <GeometryId, std::string> geometries_to_delete{};
  geometries_.swap(geometries_to_delete);

  // TODO(SeanCurtis-TRI): Mimic the full tree structure in SceneGraph.
  // SceneGraph supports arbitrary hierarchies of frames just like Meshcat.
  // This code is arbitrarily flattening it because the current SceneGraph API
  // is insufficient to support walking the tree.
  for (FrameId frame_id : inspector.GetAllFrameIds()) {
    std::string frame_path =
        frame_id == inspector.world_frame_id()
            ? params_.prefix
            : fmt::format("{}/{}", params_.prefix, inspector.GetName(frame_id));
    // MultibodyPlant declares frames with SceneGraph using "::". We replace
    // those with `/` here to expose the full tree to Meshcat.
    size_t pos = 0;
    while ((pos = frame_path.find("::", pos)) != std::string::npos) {
      frame_path.replace(pos++, 2, "/");
    }
    if (frame_id != inspector.world_frame_id() &&
        inspector.NumGeometriesForFrameWithRole(frame_id, params_.role) > 0) {
      dynamic_frames_[frame_id] = frame_path;
      frames_to_delete.erase(frame_id);  // Don't delete this one.
    }

    for (GeometryId geom_id : inspector.GetGeometries(frame_id, params_.role)) {
      // Note: We use the frame_path/id instead of instance.GetName(geom_id),
      // which is a garbled mess of :: and _ and a memory address by default
      // when coming from MultibodyPlant.
      // TODO(russt): Use the geometry names if/when they are cleaned up.
      const std::string path =
          fmt::format("{}/{}", frame_path, geom_id.get_value());
      const Rgba rgba = inspector.GetProperties(geom_id, params_.role)
          ->GetPropertyOrDefault("phong", "diffuse", params_.default_color);

      meshcat_->SetObject(path, inspector.GetShape(geom_id), rgba);
      meshcat_->SetTransform(path, inspector.GetPoseInFrame(geom_id));
      geometries_[geom_id] = path;
      colors_[geom_id] = rgba;
      geometries_to_delete.erase(geom_id);  // Don't delete this one.
    }
  }

  for (const auto& [geom_id, path] : geometries_to_delete) {
    unused(geom_id);
    meshcat_->Delete(path);
  }
  for (const auto& [frame_id, path] : frames_to_delete) {
    unused(frame_id);
    meshcat_->Delete(path);
  }
}

template <typename T>
void MeshcatVisualizer<T>::SetTransforms(
    const systems::Context<T>& context,
    const QueryObject<T>& query_object) const {
  for (const auto& [frame_id, path] : dynamic_frames_) {
    const math::RigidTransformd X_WF =
        internal::convert_to_double(query_object.GetPoseInWorld(frame_id));
    if (!recording_ || set_transforms_while_recording_) {
      meshcat_->SetTransform(path, X_WF);
    }
    if (recording_) {
      animation_->SetTransform(
          animation_->frame(ExtractDoubleOrThrow(context.get_time())), path,
          X_WF);
    }
  }
}

template <typename T>
void MeshcatVisualizer<T>::SetDeformableMeshes(
      const systems::Context<T>&, const QueryObject<T>& query_object,
      const std::vector<internal::MeshcatDeformableMeshData>& deformable_data) const {
  // Geometries registered previously that are not set again here should be
  // deleted.
  std::map<GeometryId, std::string> deformable_geometries_to_delete{};
  deformable_geometries_.swap(deformable_geometries_to_delete);

  auto color = [](int index) {
    switch(index) {
      case 0:
        return Rgba(1, 0, 0, 1);
      case 1:
        return Rgba(0, 1, 0, 1);
      case 2:
        return Rgba(0, 0, 1, 1);
      default:
        return Rgba(0.5, 0.3, 0.6, 1);
    }
  };

  for (int idx = 0; idx < static_cast<int>(deformable_data.size()); ++idx) {
    const internal::MeshcatDeformableMeshData& data = deformable_data[idx];
    const GeometryId g_id = data.geometry_id;
    const VectorX<T>& volume_vertex_positions =
        query_object.GetConfigurationsInWorld(g_id);

    /* Allocate matrices for vertices and faces for consumption by Meshcat.*/
    Eigen::Matrix3Xd vertices(3, data.surface_to_volume_vertices.size());
    Eigen::Matrix3Xi faces(3, data.surface_triangles.size());

    // Copy the surface vertex positions from the volume vertex positions
    for (int i = 0;
         i < static_cast<int>(data.surface_to_volume_vertices.size()); ++i) {
      const int v_i = data.surface_to_volume_vertices[i];
      for (int d = 0; d < 3; ++d) {
        vertices(d, i) =
            ExtractDoubleOrThrow(volume_vertex_positions[3 * v_i + d]);
      }
    }

    for (int i = 0; i < static_cast<int>(data.surface_triangles.size()); ++i) {
      faces(0, i) = data.surface_triangles[i][0];
      faces(1, i) = data.surface_triangles[i][1];
      faces(2, i) = data.surface_triangles[i][2];
    }

    const std::string path =
        fmt::format("{}/{}", params_.prefix, g_id.get_value());
    const std::string mesh_path = fmt::format("{}_{}", path, "mesh");
    const std::string wireframe_path = fmt::format("{}_{}", path, "wireframe");

    meshcat_->SetTriangleMesh(mesh_path, vertices, faces, color(idx));
    meshcat_->SetTriangleMesh(wireframe_path, vertices, faces,
                              params_.default_color, true);

    deformable_geometries_[g_id] = path;
    deformable_geometries_to_delete.erase(g_id);
  }

  for (const auto& [geom_id, path] : deformable_geometries_to_delete) {
    unused(geom_id);
    meshcat_->Delete(path);
  }
}

template <typename T>
void MeshcatVisualizer<T>::SetColorAlphas() const {
  for (const auto& [geom_id, path] : geometries_) {
    Rgba color = colors_[geom_id];
    color.set(color.r(), color.g(), color.b(), alpha_value_ * color.a());
    meshcat_->SetProperty(path, "color",
      {color.r(), color.g(), color.b(), alpha_value_ * color.a()});
  }
}

template <typename T>
systems::EventStatus MeshcatVisualizer<T>::OnInitialization(
    const systems::Context<T>&) const {
  Delete();
  return systems::EventStatus::Succeeded();
}

template <typename T>
void MeshcatVisualizer<T>::CalcMeshcatDeformableMeshData(
    const systems::Context<T>& context,
    std::vector<internal::MeshcatDeformableMeshData>* deformable_data) const {
  DRAKE_DEMAND(deformable_data != nullptr);
  deformable_data->clear();
  const auto& query_object =
      query_object_input_port().template Eval<QueryObject<T>>(context);
  const auto& inspector = query_object.inspector();
  const std::vector<GeometryId> deformable_geometries =
      inspector.GetAllDeformableGeometryIds();
  for (const auto g_id : deformable_geometries) {
    deformable_data->emplace_back(MakeMeshcatDeformableMeshData(g_id, inspector));
  }
}

template <typename T>
const std::vector<internal::MeshcatDeformableMeshData>&
MeshcatVisualizer<T>::RefreshMeshcatDeformableMeshData(const systems::Context<T>& context) const {
  // We'll need to make sure our knowledge of deformable data can get updated.
  this->get_cache_entry(deformable_data_cache_index_)
      .get_mutable_cache_entry_value(context)
      .mark_out_of_date();

  return EvalMeshcatDeformableMeshData(context);
}

template <typename T>
const std::vector<internal::MeshcatDeformableMeshData>&
MeshcatVisualizer<T>::EvalMeshcatDeformableMeshData(const systems::Context<T>& context) const {
  return this->get_cache_entry(deformable_data_cache_index_)
      .template Eval<std::vector<internal::MeshcatDeformableMeshData>>(context);
}

}  // namespace geometry
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::geometry::MeshcatVisualizer)
