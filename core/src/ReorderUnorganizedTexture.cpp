#include "rt/ReorderUnorganizedTexture.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>
#include <educelab/core/utils/Iteration.hpp>
#include <educelab/core/utils/Math.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vtkOBBTree.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

#include "rt/Logging.hpp"
#include "rt/types/MeshToVTK.hpp"
#include "rt/util/ImageConversion.hpp"

using Scalar = double;
using Vector3 = bvh::v2::Vec<Scalar, 3>;
using BBox = bvh::v2::BBox<Scalar, 3>;
using Ray = bvh::v2::Ray<Scalar, 3>;
using Triangle = bvh::v2::Tri<Scalar, 3>;
using Node = bvh::v2::Node<Scalar, 3>;
using Bvh = bvh::v2::Bvh<Node>;
using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

using namespace rt;
using namespace educelab;

namespace
{
template<typename T>
constexpr auto type_to_cvtype() -> int = delete;

template<>
constexpr auto type_to_cvtype<float>() -> int
{
    return CV_32F;
}

template<>
constexpr auto type_to_cvtype<double>() -> int
{
    return CV_64F;
}

// cv::Vec abs specialization
template <typename T, int cn>
auto abs(const cv::Vec<T, cn>& v) -> cv::Vec<T, cn>
{
    cv::Vec<T, cn> o;
    for(int i = 0; i < cn; i++) {
        o[i] = std::abs(v[i]);
    }
    return o;
}

// cv::Vec copysign specialization
template <typename T, int cn, typename T2>
auto copysign(const cv::Vec<T, cn>& mag, const T2& sgn) -> cv::Vec<T, cn>
{
    cv::Vec<T, cn> o;
    for(int i = 0; i < cn; i++) {
        o[i] = std::copysign(mag[i], sgn[i]);
    }
    return o;
}

template<typename T, int cn>
auto matmul(const cv::Mat& m, const cv::Vec<T, cn>& v) -> cv::Vec<T, cn>
{
    cv::Mat res = m * cv::Mat(v);
    auto type = type_to_cvtype<T>();
    if(type != res.type()) {
        res.convertTo(res, type);
    }
    cv::Vec<T, cn> o;
    for(int i = 0; i < cn; i++) {
        o[i] = res.at<T>(i, 0);
    }
    return o;
}

// Generate the cartesian coordinate of barycentric coordinate uvw in tri abc
auto BaryToXYZ(
    const cv::Vec3d& uvw,
    const cv::Vec3d& a,
    const cv::Vec3d& b,
    const cv::Vec3d& c) -> cv::Vec3d
{
    return uvw[0] * a + uvw[1] * b + uvw[2] * c;
}

// Check if a value is near zero
template <
    typename T,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto NearZero(T val, T eps = 1e-7) -> bool
{
    return std::abs(val) <= eps;
}

// Calculate the average pixel density of the UV map. Each face's UVs are scaled
// by the dimensions of the texture image for its chart, so a multi-chart mesh
// with differently-sized textures contributes each region at its own density.
auto ComputeUVDensity(
    const rt::Mesh& mesh,
    const rt::UVMap& uv,
    const std::vector<cv::Mat>& imgs) -> double
{
    double density{0};
    std::size_t count{0};

    // For each face
    for (std::size_t fi = 0; fi < mesh.num_faces(); ++fi) {
        const auto& face = mesh.face(fi);
        if (face.size() < 3) {
            continue;
        }

        // Skip faces without a complete UV mapping
        if (not(uv.has(fi, 0) and uv.has(fi, 1) and uv.has(fi, 2))) {
            continue;
        }

        // Skip faces whose chart has no usable image
        const auto chart = uv.get_coordinate(fi, 0).chart;
        if (chart >= imgs.size() or imgs[chart].empty()) {
            continue;
        }
        const auto maxXIdx = imgs[chart].cols - 1.0;
        const auto maxYIdx = imgs[chart].rows - 1.0;

        // Get the 3D vertices
        std::array<cv::Vec3d, 3> pts;
        for (std::size_t k = 0; k < 3; ++k) {
            const auto& v = mesh.vertex(face[k]);
            pts[k] = {v[0], v[1], v[2]};
        }

        // Get the UV coordinates for this face, in image coordinates
        std::array<cv::Vec2d, 3> uvs;
        for (std::size_t k = 0; k < 3; ++k) {
            const auto& c = uv.get_coordinate(fi, k);
            uvs[k] = {c[0] * maxXIdx, c[1] * maxYIdx};
        }

        // Update the density for each edge
        for (std::size_t idxA = 0; idxA < 3; idxA++) {
            // Next idx in the list
            const auto idxB = (idxA == 2) ? 0 : idxA + 1;

            // Calculate 2D and 3D edge lengths
            const auto edge3D = cv::norm(pts[idxB] - pts[idxA]);
            const auto edge2D = cv::norm(uvs[idxB] - uvs[idxA]);

            // Skip if one of the lengths is zero or nan
            if (NearZero(edge3D) or std::isnan(edge3D) or NearZero(edge2D) or
                std::isnan(edge2D)) {
                continue;
            }

            // Update the density
            const auto edgeDensity = edge3D / edge2D;
            count++;
            density += (edgeDensity - density) / static_cast<double>(count);
        }
    }

    return density;
}

// An oriented bounding box as vtkOBBTree reports it: one corner plus three
// edge vectors (unit direction scaled by the extent along that direction).
struct OBBResult {
    cv::Vec3d origin{0, 0, 0};
    cv::Vec3d xAxis{1, 0, 0};
    cv::Vec3d yAxis{0, 1, 0};
    cv::Vec3d zAxis{0, 0, 1};
    std::array<double, 3> size{1, 1, 1};
};

auto ComputeOBB(vtkPolyData* mesh) -> OBBResult
{
    OBBResult res;

    const auto obbTree = vtkSmartPointer<vtkOBBTree>::New();
    obbTree->ComputeOBB(
        mesh, res.origin.val, res.xAxis.val, res.yAxis.val, res.zAxis.val,
        res.size.data());

    return res;
}

// Resolve the direction ambiguity in an OBB basis using the world axes.
//
// vtkOBBTree orders its axes by extent and gives them arbitrary signs, and
// documents no handedness guarantee, so the frame create_texture_() derives
// from the OBB is unrelated to the frame the mesh arrives in. When the caller
// guarantees the input is already canonically oriented -- mesh right -> +X,
// mesh up -> +Y, surface normal -> +Z -- the world axes pin the OBB down.
//
// Deriving the required signs from the conventions downstream of here:
// create_texture_() rotates OBB x -> +X and OBB y -> -Y, then samples with u
// along -X, v along +Y, tracing rays from z_max toward -Z (the default
// !useFirstIntersection_). In terms of the *input* axes that makes
//
//     u = -obb.x,  v = -obb.y,  sampled face points along -obb.z
//
// The canonical image wants u = +X (image right = mesh right), v = -Y (image
// down = mesh down) and the +Z-facing surface sampled, so we need
//
//     obb.x = -X,  obb.y = +Y,  obb.z = -Z
//
// which is a consistent right-handed triple: (-X) x (+Y) = -Z. Only the axis
// directions change; the box, its extents and therefore the sampling geometry
// (and its foreshortening) are exactly what they were.
auto CanonicalizeOBB(const OBBResult& obb) -> OBBResult
{
    static const std::array<cv::Vec3d, 3> world{
        cv::Vec3d{1, 0, 0}, cv::Vec3d{0, 1, 0}, cv::Vec3d{0, 0, 1}};

    const std::array<cv::Vec3d, 3> edge{obb.xAxis, obb.yAxis, obb.zAxis};
    std::array<cv::Vec3d, 3> dir{};
    for (const auto i : range(3)) {
        const auto len = cv::norm(edge[i]);
        dir[i] = (len > 0.) ? cv::Vec3d(edge[i] / len) : world[i];
    }

    // Assign OBB axes to image axes by world-axis agreement rather than by
    // extent. For a near-square fragment the extent ordering can put the OBB
    // axes on the opposite world axes, a 90-degree error that no amount of
    // sign correction can undo.
    std::array<std::size_t, 3> pick{0, 1, 2};
    std::array<bool, 3> used{false, false, false};
    for (const auto axis : range(2)) {
        std::size_t best{0};
        double bestDot{-1.};
        for (const auto i : range(3)) {
            if (used[i]) {
                continue;
            }
            if (const auto d = std::abs(dir[i].dot(world[axis])); d > bestDot) {
                bestDot = d;
                best = static_cast<std::size_t>(i);
            }
        }
        pick[axis] = best;
        used[best] = true;
    }
    const auto unused = std::find(used.begin(), used.end(), false);
    pick[2] = static_cast<std::size_t>(std::distance(used.begin(), unused));

    std::array<cv::Vec3d, 3> ax{edge[pick[0]], edge[pick[1]], edge[pick[2]]};

    // Flip the in-plane axes to the signs derived above. Negating an edge
    // vector moves the box corner to the far end of that edge, so the origin
    // has to follow or the result no longer describes the same box.
    OBBResult out;
    out.origin = obb.origin;
    const std::array<double, 2> wantSign{-1., 1.};  // x . X < 0, y . Y > 0
    for (const auto i : range(2)) {
        if (ax[i].dot(world[i]) * wantSign[i] < 0.) {
            out.origin += ax[i];
            ax[i] = -ax[i];
        }
    }

    // Complete the triple as right-handed, putting the third axis on -Z. Flip
    // the edge vtkOBBTree gave us rather than substituting the cross product,
    // so the axis keeps its exact extent and orthogonality.
    //
    // This is bookkeeping as far as create_texture_() is concerned: it builds
    // its realignment from the first two axes alone and consumes zAxis only in
    // the (flip-invariant) centroid translation, before overwriting it from the
    // realigned AABB. Which face gets sampled is decided by that realignment
    // being a proper rotation, not by this block -- see AlignVectorToVector.
    // Kept so the returned box is self-consistent for callers that do read it.
    if (ax[2].dot(ax[0].cross(ax[1])) < 0.) {
        out.origin += ax[2];
        ax[2] = -ax[2];
    }

    out.xAxis = ax[0];
    out.yAxis = ax[1];
    out.zAxis = ax[2];
    // Permuted with the axes to keep size[i] describing axis i. That is the
    // invariant worth holding; VTK's extra guarantee that the list is sorted
    // descending was only a side effect of its axes being extent-ordered.
    out.size = {obb.size[pick[0]], obb.size[pick[1]], obb.size[pick[2]]};
    return out;
}

// Compute the OBB a sampling frame is derived from.
//
// The OBB's axis directions are arbitrary, which leaves the orientation of the
// output image unrelated to the orientation of the input mesh. Under
// OrientationMode::Canonical the caller guarantees a canonically oriented
// input, so the world axes pin the box down first. Every frame-deriving path
// wants both halves of this, and doing only the first silently reintroduces
// the arbitrary orientation.
auto ComputeOBB(
    vtkPolyData* mesh,
    const ReorderUnorganizedTexture::OrientationMode orientation) -> OBBResult
{
    auto obb = ComputeOBB(mesh);
    if (orientation == ReorderUnorganizedTexture::OrientationMode::Canonical) {
        obb = CanonicalizeOBB(obb);
    }
    return obb;
}

auto AlignVectorToVector(cv::Vec3d a, const cv::Vec3d& b, const cv::Vec3d& c) -> cv::Mat
{
    // Identity initial rotation
    cv::Mat r = cv::Mat::eye(3, 3, CV_64F);

    // First, check for anti-parallel vectors at low precision
    if (almost_equal(a.dot(b), -1., 1e-3)) {
        auto d = abs(c) - cv::Vec3d{1., 1., 1.};
        d = copysign(cv::Vec3d{1., 1., 1.}, d);
        // Write the diagonal element-wise. Assigning a Vec to Mat::diag()
        // instead converts it to a cv::Scalar and fills the whole diagonal
        // with d[0], turning this 180-degree rotation about the c axis into a
        // reflection (det = -1) that silently mirrors the sampled surface.
        cv::Mat(d, false).copyTo(r.diag());
    }

    // Next, return early for parallel vectors at high precision
    a = matmul(r, a);
    if (almost_equal(a.dot(b), 1.)) {
        return r;
    }

    // Finally, refine the rotation with a special sform of Rodriguez rotation
    // https://math.stackexchange.com/a/476311
    auto v = a.cross(b);
    const auto c2 = a.dot(b);
    const cv::Mat vx = (cv::Mat_<double>(3,3) << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0);

    const cv::Mat rod = cv::Mat::eye(3, 3, CV_64F) + vx + vx * vx / (1 + c2);
    return rod * r;
}

auto MinimizeAABB(vtkPolyData* mesh)
{
    const auto tfm = vtkSmartPointer<vtkTransform>::New();
    const auto apply = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    apply->SetTransform(tfm);
    apply->SetInputData(mesh);

    double minAngle{0}, minArea{INF<double>};
    std::array<double, 6> bbox{};
    for (const auto angle : range(-45., 45.5, 0.5)) {
        tfm->Identity();
        tfm->RotateZ(angle);
        apply->Update();

        const vtkSmartPointer tmp = apply->GetOutput();
        tmp->ComputeBounds();
        tmp->GetBounds(bbox.data());

        if (const auto area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]);
            area < minArea) {
            minAngle = angle;
            minArea = area;
        }
    }

    // Return final result
    tfm->Identity();
    tfm->RotateZ(minAngle);
    apply->Update();
    vtkSmartPointer result = apply->GetOutput();

    return result;
}

// Generate a new UV map using the aligned mesh
// This is simple after alignment u = pos.x / max.x, v = pos.y / max.y
auto CreateUVMap(
    vtkPolyData* mesh,
    const cv::Vec3d& o,
    const cv::Vec3d& x,
    const cv::Vec3d& y) -> rt::UVMap
{
    rt::UVMap out;

    const auto uLen = cv::norm(x);
    const auto vLen = cv::norm(y);
    const auto uVec = x / uLen;
    const auto vVec = y / vLen;

    // Add points to the pool (one UV per vertex; pool index == point index)
    cv::Vec3d p;
    for (const auto ptID : range(mesh->GetNumberOfPoints())) {
        mesh->GetPoint(ptID, p.val);

        auto u = (p - o).dot(uVec) / uLen;
        auto v = (p - o).dot(vVec) / vLen;

        [[maybe_unused]] const auto idx =
            out.insert(static_cast<float>(u), static_cast<float>(v));
    }

    // Map per-wedge UVs (vertex index doubles as pool index)
    const auto ptIDs = vtkSmartPointer<vtkIdList>::New();
    for (const auto cellIdx : range(mesh->GetNumberOfCells())) {
        mesh->GetCellPoints(cellIdx, ptIDs);
        std::size_t corner{0};
        for (const auto ptID : *ptIDs) {
            out.map(cellIdx, corner++, static_cast<std::size_t>(ptID));
        }
    }

    return out;
}

auto IntersectRay(Ray ray, const Bvh& bvh, const std::vector<PrecomputedTri>& tris)
{
    struct HitRecord {
        std::size_t primitiveIdx;
        Scalar distance;
        struct
        {
            Scalar u;
            Scalar v;
        } intersection;
    };
    using ReturnType = std::optional<HitRecord>;

    static constexpr auto invalidID = std::numeric_limits<std::size_t>::max();
    static constexpr std::size_t stack_size = 64;
    static constexpr bool use_robust_traversal = true;

    auto primId = invalidID;
    Scalar u, v;

    // Traverse the BVH and get the u, v coordinates of the closest intersection.
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    bvh.intersect<false, use_robust_traversal>(ray, bvh.get_root().index, stack,
        [&] (const auto begin, const auto end) {
            for (auto i = begin; i < end; ++i) {
                if (auto hit = tris[i].intersect(ray)) {
                    primId = i;
                    u = hit.value().first;
                    v = hit.value().second;
                }
            }
            return primId != invalidID;
        });

    if (primId == invalidID) {
        return ReturnType();
    }

    HitRecord hit{primId, ray.tmax, {u, v}};
    return std::make_optional(hit);
}

// Build a BVH over the mesh's triangles. precompTris is reordered to match
// bvh.prim_ids, matching the traversal in IntersectRay().
struct BVHData {
    Bvh bvh;
    std::vector<PrecomputedTri> precompTris;
};

auto BuildBVH(vtkPolyData* mesh) -> BVHData
{
    std::vector<Triangle> tris;
    auto ptIDs = vtkSmartPointer<vtkIdList>::New();
    Vector3 a, b, c;
    for (auto cellIdx : range(mesh->GetNumberOfCells())) {
        mesh->GetCellPoints(cellIdx, ptIDs);
        mesh->GetPoint(ptIDs->GetId(0), a.values);
        mesh->GetPoint(ptIDs->GetId(1), b.values);
        mesh->GetPoint(ptIDs->GetId(2), c.values);
        tris.emplace_back(a, b, c);
    }

    bvh::v2::ThreadPool threadPool;
    bvh::v2::ParallelExecutor executor(threadPool);

    std::vector<BBox> bboxes(tris.size());
    std::vector<Vector3> centers(tris.size());
    executor.for_each(0, tris.size(), [&](const auto begin, const auto end) {
        for (auto i = begin; i < end; ++i) {
            bboxes[i] = tris[i].get_bbox();
            centers[i] = tris[i].get_center();
        }
    });

    bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
    auto bvh =
        bvh::v2::DefaultBuilder<Node>::build(threadPool, bboxes, centers, config);

    std::vector<PrecomputedTri> precompTris(tris.size());
    executor.for_each(0, tris.size(), [&](const auto begin, const auto end) {
        for (auto i = begin; i < end; ++i) {
            auto j = bvh.prim_ids[i];
            precompTris[i] = tris[j];
        }
    });

    return BVHData{std::move(bvh), std::move(precompTris)};
}

// Derive a sensible pinhole camera that frames the (roughly planar) mesh,
// looking along its shortest OBB axis at the centroid. @c sampleRate is the
// target surface sampling in mesh units per pixel (e.g. the input texture's
// pixel density); pass <= 0 to fall back to a fixed long-edge size.
auto AutoCamera(
    vtkPolyData* mesh,
    double sampleRate,
    ReorderUnorganizedTexture::OrientationMode orientation)
    -> ReorderUnorganizedTexture::ProjectionParams
{
    using OrientationMode = ReorderUnorganizedTexture::OrientationMode;

    const auto obb = ComputeOBB(mesh, orientation);
    const auto& [origin, xAxis, yAxis, zAxis, size] = obb;
    const cv::Vec3d centroid = origin + 0.5 * (xAxis + yAxis + zAxis);
    const auto ex = cv::norm(xAxis);
    const auto ey = cv::norm(yAxis);
    cv::Vec3d nrm = cv::normalize(zAxis);  // shortest (thin) axis
    if (orientation == OrientationMode::Canonical) {
        // CanonicalizeOBB resolves the thin axis onto world -Z, which is what
        // the orthographic sampling frame wants. A camera wants the opposite:
        // it belongs on the +Z side, looking back at the surface that faces
        // the viewer.
        nrm = -nrm;
    }

    // Place the camera off the surface for mild perspective
    auto d = 1.5 * std::max(ex, ey);
    if (d <= 0) {
        d = 1.0;
    }
    const cv::Vec3d eye = centroid + nrm * d;

    // Size the output to approximately preserve the input texture pixel
    // density. At distance d a pixel subtends ~d/f mesh units on the surface,
    // so f = d / sampleRate matches the texture density and the mesh spans
    // ~extent / sampleRate pixels. Fall back to a fixed long edge when no
    // usable density is available.
    int width{0};
    int height{0};
    double f{0};
    if (sampleRate > 0 and std::isfinite(sampleRate)) {
        constexpr double margin = 1.05;
        width =
            std::max(1, static_cast<int>(std::ceil(ex / sampleRate * margin)));
        height =
            std::max(1, static_cast<int>(std::ceil(ey / sampleRate * margin)));
        f = d / sampleRate;
    } else {
        constexpr int kMaxDim = 2048;
        width = kMaxDim;
        height = kMaxDim;
        if (ex >= ey) {
            height =
                std::max(1, static_cast<int>(std::round(kMaxDim * ey / ex)));
        } else {
            width =
                std::max(1, static_cast<int>(std::round(kMaxDim * ex / ey)));
        }
        f = std::min(width * d / (ex * 1.1), height * d / (ey * 1.1));
    }

    // Look-at, OpenCV convention (+Z forward into scene, +Y down).
    // Under Canonical the box gives forward = -Z and worldUp = +Y, hence
    // right = (-Z) x (+Y) = +X and down = (-Z) x (+X) = -Y: image right along
    // world +X and image down along world -Y, the same convention the
    // orthographic path produces.
    const cv::Vec3d forward = cv::normalize(centroid - eye);
    const cv::Vec3d worldUp = cv::normalize(yAxis);
    const cv::Vec3d right = cv::normalize(forward.cross(worldUp));
    const cv::Vec3d down = forward.cross(right);

    ReorderUnorganizedTexture::ProjectionParams p;
    p.fx = f;
    p.fy = f;
    p.cx = width / 2.0;
    p.cy = height / 2.0;
    p.width = width;
    p.height = height;
    const std::array<cv::Vec3d, 3> rows{right, down, forward};
    const cv::Vec3d t{
        -right.dot(eye), -down.dot(eye), -forward.dot(eye)};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            p.extrinsics(i, j) = rows[i][j];
        }
        p.extrinsics(i, 3) = t[i];
    }
    return p;
}

// Generate a UV map by projecting each vertex through the pinhole camera.
// Vertices behind the camera get sentinel (-1, -1) coordinates.
// Limitation: no near-plane clipping. A triangle that straddles the camera
// plane (some vertices in front, some behind) keeps its behind-camera vertices
// at the sentinel UV, so that face will texture-map incorrectly.
auto CreateProjectiveUVMap(
    vtkPolyData* mesh, const ReorderUnorganizedTexture::ProjectionParams& cam)
    -> rt::UVMap
{
    rt::UVMap out;
    cv::Matx33d R;
    cv::Vec3d t;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R(i, j) = cam.extrinsics(i, j);
        }
        t[i] = cam.extrinsics(i, 3);
    }
    const auto maxX = cam.width - 1.0;
    const auto maxY = cam.height - 1.0;

    cv::Vec3d p;
    for (const auto ptID : range(mesh->GetNumberOfPoints())) {
        mesh->GetPoint(ptID, p.val);
        const cv::Vec3d pc = R * p + t;
        if (pc[2] <= 0) {
            [[maybe_unused]] const auto idx = out.insert(-1.0F, -1.0F);
            continue;
        }
        // Project to normalized image coords, apply radial distortion, then
        // scale by the focal length and offset by the principal point.
        const cv::Vec2d dist =
            rt::DistortNormalized(cam, {pc[0] / pc[2], pc[1] / pc[2]});
        const auto px = cam.fx * dist[0] + cam.cx;
        const auto py = cam.fy * dist[1] + cam.cy;
        [[maybe_unused]] const auto idx = out.insert(
            static_cast<float>(px / maxX), static_cast<float>(py / maxY));
    }

    const auto ptIDs = vtkSmartPointer<vtkIdList>::New();
    for (const auto cellIdx : range(mesh->GetNumberOfCells())) {
        mesh->GetCellPoints(cellIdx, ptIDs);
        std::size_t corner{0};
        for (const auto ptID : *ptIDs) {
            out.map(cellIdx, corner++, static_cast<std::size_t>(ptID));
        }
    }
    return out;
}

}  // namespace

void ReorderUnorganizedTexture::setMesh(const Mesh::Pointer& mesh)
{
    inputMesh_ = mesh;
}

void ReorderUnorganizedTexture::setUVMap(const rt::UVMap& uv) { inputUV_ = uv; }

void ReorderUnorganizedTexture::setTextureMats(const std::vector<cv::Mat>& imgs)
{
    // Normalize each image to 8-bit, 3-channel (BGR). Empty images are kept in
    // place so the vector stays indexable by UV chart. See setTextureMats() docs
    // for the bit-depth/channel-support caveat.
    inputTextures_.clear();
    inputTextures_.reserve(imgs.size());
    for (const auto& img : imgs) {
        if (img.empty()) {
            inputTextures_.emplace_back();
            continue;
        }
        auto out = rt::QuantizeImage(img, CV_8U);
        out = rt::ColorConvertImage(out, 3);
        inputTextures_.push_back(std::move(out));
    }
}

void ReorderUnorganizedTexture::setSamplingOrigin(const SamplingOrigin o)
{
    sampleOrigin_ = o;
}

auto ReorderUnorganizedTexture::samplingOrigin() const -> SamplingOrigin
{
    return sampleOrigin_;
}

void ReorderUnorganizedTexture::setSamplingMode(const SamplingMode m)
{
    sampleMode_ = m;
}

auto ReorderUnorganizedTexture::samplingMode() const -> SamplingMode
{
    return sampleMode_;
}

void ReorderUnorganizedTexture::setSampleRate(const double s)
{
    sampleRate_ = s;
}

auto ReorderUnorganizedTexture::sampleRate() const -> double
{
    return sampleRate_;
}

void ReorderUnorganizedTexture::setSampleDim(const std::size_t d)
{
    sampleDim_ = d;
}

auto ReorderUnorganizedTexture::sampleDim() const -> std::size_t
{
    return sampleDim_;
}

void ReorderUnorganizedTexture::setUseFirstIntersection(const bool b)
{
    useFirstIntersection_ = b;
}

auto ReorderUnorganizedTexture::useFirstIntersection() const -> bool
{
    return useFirstIntersection_;
}

void ReorderUnorganizedTexture::setProjectionMode(const ProjectionMode m)
{
    projectionMode_ = m;
}

auto ReorderUnorganizedTexture::projectionMode() const -> ProjectionMode
{
    return projectionMode_;
}

void ReorderUnorganizedTexture::setOrientationMode(const OrientationMode m)
{
    orientationMode_ = m;
}

auto ReorderUnorganizedTexture::orientationMode() const -> OrientationMode
{
    return orientationMode_;
}

void ReorderUnorganizedTexture::setProjectionParams(const ProjectionParams& params)
{
    projParams_ = params;
    projParamsSet_ = true;
}

auto ReorderUnorganizedTexture::projectionParams() const -> ProjectionParams
{
    return projParams_;
}

void ReorderUnorganizedTexture::clearProjectionParams()
{
    projParams_ = {};
    projParamsSet_ = false;
}

auto rt::ValidateProjectionParams(
    const ReorderUnorganizedTexture::ProjectionParams& p)
    -> std::optional<std::string>
{
    if (p.width <= 0 or p.height <= 0) {
        return "image size (width, height) must be positive";
    }
    if (not std::isfinite(p.fx) or not std::isfinite(p.fy) or p.fx <= 0.0 or
        p.fy <= 0.0) {
        return "focal lengths (fx, fy) must be positive and finite";
    }
    if (not std::isfinite(p.cx) or not std::isfinite(p.cy)) {
        return "principal point (cx, cy) must be finite";
    }
    if (not std::isfinite(p.k1) or not std::isfinite(p.k2) or
        not std::isfinite(p.k3)) {
        return "radial distortion coefficients (k1, k2, k3) must be finite";
    }

    // Rotation block of the world-to-camera extrinsics must be a proper
    // rotation: orthonormal (R*R^T == I) and right-handed (det(R) == +1).
    cv::Matx33d r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r(i, j) = p.extrinsics(i, j);
        }
    }
    constexpr double eps{1e-6};
    const cv::Matx33d rrt = r * r.t();
    const cv::Matx33d eye = cv::Matx33d::eye();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (std::abs(rrt(i, j) - eye(i, j)) > eps) {
                return "extrinsics rotation block must be orthonormal";
            }
        }
    }
    if (std::abs(cv::determinant(r) - 1.0) > eps) {
        return "extrinsics rotation block must be right-handed (det = +1)";
    }

    return std::nullopt;
}

auto rt::DistortNormalized(
    const ReorderUnorganizedTexture::ProjectionParams& p,
    const cv::Vec2d& normalized) -> cv::Vec2d
{
    const auto r2 = normalized.dot(normalized);
    const auto rad = 1.0 + p.k1 * r2 + p.k2 * r2 * r2 + p.k3 * r2 * r2 * r2;
    return normalized * rad;
}

auto rt::UndistortNormalized(
    const ReorderUnorganizedTexture::ProjectionParams& p,
    const cv::Vec2d& distorted) -> cv::Vec2d
{
    // Identity fast path when there's no distortion
    if (p.k1 == 0.0 and p.k2 == 0.0 and p.k3 == 0.0) {
        return distorted;
    }
    // No closed form; iterate (OpenCV undistortPoints). Five iterations is
    // plenty for the small coefficients seen in practice.
    cv::Vec2d u = distorted;
    for (int i = 0; i < 5; ++i) {
        const auto r2 = u.dot(u);
        const auto inv =
            1.0 / (1.0 + p.k1 * r2 + p.k2 * r2 * r2 + p.k3 * r2 * r2 * r2);
        u = distorted * inv;
    }
    return u;
}

auto ReorderUnorganizedTexture::getUVMap() -> rt::UVMap { return outputUV_; }

auto ReorderUnorganizedTexture::getDepthMap() -> cv::Mat
{
    return outputDepthMap_;
}

auto ReorderUnorganizedTexture::getPositionMap() -> cv::Mat
{
    return outputPositionMap_;
}

// Compute the result
auto ReorderUnorganizedTexture::compute() -> cv::Mat
{
    if (projectionMode_ == ProjectionMode::Camera) {
        create_texture_camera_();
    } else {
        create_texture_();
    }
    return outputTexture_;
}

void ReorderUnorganizedTexture::create_texture_()
{
    // Compute mesh's (rough) OBB, pinned to the world axes under Canonical
    auto mesh = rt::MeshToVTK(*inputMesh_);
    auto obb = ComputeOBB(mesh, orientationMode_);
    auto [origin, xAxis, yAxis, zAxis, size] = obb;

    // We're going to transform the mesh to be axis-aligned. Not strictly
    // necessary, but lets us more easily minimize the XY area of the OBB
    auto tfm = vtkSmartPointer<vtkTransform>::New();
    tfm->PostMultiply();

    // Transform for moving the centroid to the origin
    auto t = -(origin + 0.5 * (xAxis + yAxis + zAxis));
    tfm->Translate(t[0], t[1], t[2]);

    // Compute rotation from OBB +X and +Y axes to the basis +X and -Y
    auto xAxNorm = cv::normalize(xAxis);
    auto yAxNorm = cv::normalize(yAxis);
    auto rotX = AlignVectorToVector(xAxNorm, {1, 0, 0}, {0, -1, 0});
    auto rotY =
        AlignVectorToVector(matmul(rotX, yAxNorm), {0, -1, 0}, {1, 0, 0});
    cv::Mat rot = rotY * rotX;

    // Convert rotation to vtkTransform
    std::array<double, 16> m{};
    for (const auto [row, col] : range2D(3, 3)) {
        auto flat = row * 4 + col;
        m[flat] = rot.at<double>(row, col);
    }
    auto rotation = vtkSmartPointer<vtkTransform>::New();
    rotation->SetMatrix(m.data());
    tfm->Concatenate(rotation);

    // Apply the reorientation transform
    auto apply = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    apply->SetTransform(tfm);
    apply->SetInputData(mesh);
    apply->Update();

    // Rotate around Z to minimize the XY area of the AABB
    mesh = apply->GetOutput();
    mesh = MinimizeAABB(mesh);

    // Update our sampling bounds using the new AABB
    // Note: X-axis is flipped to mimic the basis returned by OBBTree
    std::array<double, 6> bbox{};
    mesh->ComputeBounds();
    mesh->GetBounds(bbox.data());
    origin = {bbox[1], bbox[2], bbox[4]};
    xAxis = {bbox[0] - bbox[1], 0., 0.};
    yAxis = {0., bbox[3] - bbox[2], 0.};
    zAxis = {0., 0., bbox[5] - bbox[4]};

    // Create BVH for mesh
    auto bvhData = BuildBVH(mesh);
    auto& bvh = bvhData.bvh;
    auto& precompTris = bvhData.precompTris;

    // Texture init
    int cols{-1};
    int rows{-1};
    auto xLen = cv::norm(xAxis);
    auto yLen = cv::norm(yAxis);

    // Calculate the sample rate
    double sampleRate{DEFAULT_SAMPLE_RATE};
    switch (sampleMode_) {
        case SamplingMode::Rate:
            sampleRate = sampleRate_;
            cols = static_cast<int>(std::ceil(xLen / sampleRate));
            rows = static_cast<int>(std::ceil(yLen / sampleRate));
            break;
        case SamplingMode::OutputWidth:
            sampleRate = xLen / static_cast<double>(sampleDim_);
            cols = static_cast<int>(sampleDim_);
            rows = static_cast<int>(std::ceil(yLen / sampleRate));
            break;
        case SamplingMode::OutputHeight:
            sampleRate = yLen / static_cast<double>(sampleDim_);
            cols = static_cast<int>(std::ceil(xLen / sampleRate));
            rows = static_cast<int>(sampleDim_);
            break;
        case SamplingMode::AutoUV:
            sampleRate =
                ::ComputeUVDensity(*inputMesh_, inputUV_, inputTextures_);
            cols = static_cast<int>(std::ceil(xLen / sampleRate));
            rows = static_cast<int>(std::ceil(yLen / sampleRate));
            break;
    }

    logger()->debug(
        "Output size: {}x{} (Sample rate: {:.5g})", cols, rows, sampleRate);

    // Set up the output image. Depth/position default to NaN so pixels with no
    // surface intersection are distinguishable from valid zero-valued samples.
    constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
    outputTexture_ = cv::Mat::zeros(rows, cols, CV_8UC3);
    outputDepthMap_ = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(kNaN));
    outputPositionMap_ =
        cv::Mat(rows, cols, CV_32FC3, cv::Scalar(kNaN, kNaN, kNaN));

    // Normalize the length
    auto normedX = cv::normalize(xAxis);
    auto normedY = cv::normalize(yAxis);
    auto zLen = cv::norm(zAxis);
    if (zLen < 1.0) {
        zAxis = normedY.cross(normedX);
        zLen = cv::norm(zAxis);
    } else {
        zAxis = cv::normalize(zAxis);
    }

    // Change the origin and axes directions w.r.t. the sampling origin
    switch (sampleOrigin_) {
        case SamplingOrigin::TopLeft:
            // Already setup. Do nothing.
            break;
        case SamplingOrigin::TopRight:
            origin += xLen * normedX;
            normedX *= -1;
            break;
        case SamplingOrigin::BottomLeft:
            origin += yLen * normedY;
            normedY *= -1;
            break;
        case SamplingOrigin::BottomRight:
            origin += xLen * normedX + yLen * normedY;
            normedX *= -1;
            normedY *= -1;
            break;
    }

    const bool haveTexture = std::any_of(
        inputTextures_.begin(), inputTextures_.end(),
        [](const cv::Mat& m) { return not m.empty(); });
    missingCharts_.clear();
    for (auto [v, u] : range2D(rows, cols)) {
        // Sample through the pixel center to avoid a half-pixel bias
        auto uOffset = (u + 0.5) * sampleRate * normedX;
        auto vOffset = (v + 0.5) * sampleRate * normedY;

        // Get t
        auto a0 = origin + uOffset + vOffset;
        auto a1 = zAxis;
        if (not useFirstIntersection_) {
            a0 = a0 + zAxis * zLen;
            a1 *= -1;
        }

        // Intersect a ray with the data structure
        Vector3 start(a0[0], a0[1], a0[2]);
        Vector3 dir(a1[0], a1[1], a1[2]);
        Ray ray(start, dir, 0.0, zLen * 2);
        auto hit = IntersectRay(ray, bvh, precompTris);
        if (not hit) {
            continue;
        }

        // The ray direction is unit length, so distance is perpendicular depth
        const auto dist = hit.value().distance;
        outputDepthMap_.at<float>(v, u) = static_cast<float>(dist);

        // 3D surface position (in the realigned sampling frame)
        const cv::Vec3d pos = a0 + a1 * dist;
        outputPositionMap_.at<cv::Vec3f>(v, u) = cv::Vec3f(
            static_cast<float>(pos[0]), static_cast<float>(pos[1]),
            static_cast<float>(pos[2]));

        // Sample the surface color into the output texture
        if (haveTexture) {
            const auto cellId = bvh.prim_ids[hit.value().primitiveIdx];
            if (const auto* img = resolve_chart_image_(cellId)) {
                const auto inter = hit.value().intersection;
                outputTexture_.at<cv::Vec3b>(v, u) =
                    sample_surface_color_(*img, cellId, inter.u, inter.v);
            }
        }
    }
    report_missing_charts_();

    outputUV_ = CreateUVMap(mesh, origin, xAxis, yAxis);
}

void ReorderUnorganizedTexture::create_texture_camera_()
{
    // Sample the mesh in its native (world) frame; no realignment.
    auto mesh = rt::MeshToVTK(*inputMesh_);

    // Resolve camera parameters (auto-derive if not explicitly provided). The
    // auto camera is sized to preserve the input texture's pixel density.
    const bool haveTexture = std::any_of(
        inputTextures_.begin(), inputTextures_.end(),
        [](const cv::Mat& m) { return not m.empty(); });
    double texDensity{0.0};
    if (haveTexture) {
        texDensity = ::ComputeUVDensity(*inputMesh_, inputUV_, inputTextures_);
    }
    const ProjectionParams cam =
        projParamsSet_ ? projParams_
                       : ::AutoCamera(mesh, texDensity, orientationMode_);
    if (const auto err = rt::ValidateProjectionParams(cam)) {
        throw std::runtime_error("Camera projection: " + *err);
    }

    // Build the BVH over the world-frame mesh
    auto bvhData = ::BuildBVH(mesh);
    auto& bvh = bvhData.bvh;
    auto& precompTris = bvhData.precompTris;

    const int cols = cam.width;
    const int rows = cam.height;
    constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
    outputTexture_ = cv::Mat::zeros(rows, cols, CV_8UC3);
    outputDepthMap_ = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(kNaN));
    outputPositionMap_ =
        cv::Mat(rows, cols, CV_32FC3, cv::Scalar(kNaN, kNaN, kNaN));

    // Decompose world->camera extrinsics: x_cam = R * X_world + t
    cv::Matx33d R;
    cv::Vec3d t;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R(i, j) = cam.extrinsics(i, j);
        }
        t[i] = cam.extrinsics(i, 3);
    }
    const cv::Matx33d rInv = R.t();
    const cv::Vec3d camCenter = -rInv * t;

    // Far clip from the camera-to-scene distance plus the mesh diagonal
    std::array<double, 6> bbox{};
    mesh->ComputeBounds();
    mesh->GetBounds(bbox.data());
    const cv::Vec3d bbMin(bbox[0], bbox[2], bbox[4]);
    const cv::Vec3d bbMax(bbox[1], bbox[3], bbox[5]);
    const auto diag = cv::norm(bbMax - bbMin);
    const auto far = (cv::norm(camCenter - 0.5 * (bbMin + bbMax)) + diag) * 2.0;

    missingCharts_.clear();
    for (auto [v, u] : range2D(rows, cols)) {
        // Pinhole ray through the pixel center: dir = R^-1 * K^-1 * [u, v, 1].
        // Undistort the normalized coords first so the ray matches the ideal
        // pinhole direction (no-op when the camera has no distortion).
        const cv::Vec2d ideal = rt::UndistortNormalized(
            cam, {(static_cast<double>(u) + 0.5 - cam.cx) / cam.fx,
                  (static_cast<double>(v) + 0.5 - cam.cy) / cam.fy});
        const cv::Vec3d dCam(ideal[0], ideal[1], 1.0);
        const auto dCamNorm = cv::norm(dCam);
        const cv::Vec3d dWorld = rInv * dCam / dCamNorm;  // unit ray direction

        Vector3 start(camCenter[0], camCenter[1], camCenter[2]);
        Vector3 dir(dWorld[0], dWorld[1], dWorld[2]);
        Ray ray(start, dir, 0.0, far);

        // Nearest intersection is the visible surface (handles occlusion)
        auto hit = IntersectRay(ray, bvh, precompTris);
        if (not hit) {
            continue;
        }

        // distance is along the unit ray (slant range); divide by the ray
        // obliquity to store perpendicular (optical-axis) depth = camera-space Z
        const auto dist = hit.value().distance;
        outputDepthMap_.at<float>(v, u) = static_cast<float>(dist / dCamNorm);

        // 3D surface position in the mesh's world frame
        const cv::Vec3d pos = camCenter + dWorld * dist;
        outputPositionMap_.at<cv::Vec3f>(v, u) = cv::Vec3f(
            static_cast<float>(pos[0]), static_cast<float>(pos[1]),
            static_cast<float>(pos[2]));

        // Sample the surface color into the output texture
        if (haveTexture) {
            const auto cellId = bvh.prim_ids[hit.value().primitiveIdx];
            if (const auto* img = resolve_chart_image_(cellId)) {
                const auto inter = hit.value().intersection;
                outputTexture_.at<cv::Vec3b>(v, u) =
                    sample_surface_color_(*img, cellId, inter.u, inter.v);
            }
        }
    }
    report_missing_charts_();

    outputUV_ = ::CreateProjectiveUVMap(mesh, cam);
}

auto ReorderUnorganizedTexture::resolve_chart_image_(
    const std::size_t cellId) const -> const cv::Mat*
{
    // The face's texture is the chart carried by its corner-0 UV coordinate,
    // matching how libcore groups faces by chart on write.
    const auto chart = inputUV_.get_coordinate(cellId, 0).chart;
    if (chart >= inputTextures_.size() or inputTextures_[chart].empty()) {
        missingCharts_.push_back(chart);
        return nullptr;
    }
    return &inputTextures_[chart];
}

auto ReorderUnorganizedTexture::sample_surface_color_(
    const cv::Mat& img, const std::size_t cellId, const double interU,
    const double interV) const -> cv::Vec3b
{
    // Precondition: reorder UV maps map every corner of every triangle. Both
    // CreateUVMap and CreateProjectiveUVMap insert one coordinate per cell
    // corner (behind-camera vertices get a sentinel, but are still mapped), so
    // a fully-mapped face is guaranteed for any cellId the sampler visits.
    assert(
        inputUV_.has(cellId, 0) and inputUV_.has(cellId, 1) and
        inputUV_.has(cellId, 2) &&
        "sample_surface_color_: face has an unmapped per-wedge UV corner");

    // Get the face's UV coordinates (per-wedge, in corner order)
    std::vector<cv::Vec3d> uvPts;
    for (std::size_t corner = 0; corner < 3; ++corner) {
        const auto& uv = inputUV_.get_coordinate(cellId, corner);
        uvPts.emplace_back(uv[0], uv[1], 0.0);
    }

    // Get the UV position of the intersection point.
    // Inexplicably, bvh barycentric coordinates are relative to the 2nd pt.
    const cv::Vec3d bCoord{interU, interV, 1 - interU - interV};
    const auto cPoint = ::BaryToXYZ(bCoord, uvPts[1], uvPts[2], uvPts[0]);

    // Convert the UV position to pixel coordinates (in the chart's image)
    const auto x = static_cast<float>(cPoint[0] * (img.cols - 1));
    const auto y = static_cast<float>(cPoint[1] * (img.rows - 1));

    // Bilinear interpolate color
    cv::Mat subRect;
    cv::getRectSubPix(img, {1, 1}, {x, y}, subRect);
    return subRect.at<cv::Vec3b>(0, 0);
}

void ReorderUnorganizedTexture::report_missing_charts_() const
{
    if (missingCharts_.empty()) {
        return;
    }
    std::sort(missingCharts_.begin(), missingCharts_.end());
    missingCharts_.erase(
        std::unique(missingCharts_.begin(), missingCharts_.end()),
        missingCharts_.end());

    std::string list;
    for (std::size_t i = 0; i < missingCharts_.size(); ++i) {
        list += (i == 0 ? "" : ", ") + std::to_string(missingCharts_[i]);
    }
    logger()->warn(
        "No usable texture image for UV chart(s) {}; those surface regions were "
        "left uncolored in the output texture",
        list);
    missingCharts_.clear();
}
