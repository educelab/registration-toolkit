#include "rt/ReorderUnorganizedTexture.hpp"

#include <array>
#include <cmath>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vtkOBBTree.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <educelab/core/utils/Math.hpp>
#include <educelab/core/utils/Iteration.hpp>

#include "rt/Logging.hpp"
#include "rt/types/ITK2VTK.hpp"

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

// Get the vertices belong to a cell
template <typename CellIterator>
auto GetCellVertices(const ITKMesh::Pointer& mesh, CellIterator& cell)
{
    std::vector<cv::Vec3d> pts;
    for (const auto& id : cell->Value()->GetPointIdsContainer()) {
        auto p = mesh->GetPoint(id);
        pts.emplace_back(p[0], p[1], p[2]);
    }
    return pts;
}

// Check if a value is near zero
template <
    typename T,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
auto NearZero(T val, T eps = 1e-7) -> bool
{
    return std::abs(val) <= eps;
}

// Calculate the pixel density of the UV map
auto ComputeUVDensity(
    const ITKMesh::Pointer& mesh,
    const UVMap& uv,
    const double imgWidth,
    const double imgHeight) -> double
{
    double density{0};
    std::size_t count{0};

    auto maxXIdx = imgWidth - 1;
    auto maxYIdx = imgHeight - 1;

    // For each face
    for (auto cell = mesh->GetCells()->Begin(); cell != mesh->GetCells()->End();
         ++cell) {

        // Get the 3D vertices
        auto pts = ::GetCellVertices(mesh, cell);

        // Get the UV coordinates for this face
        auto uvs = uv.getFaceUVs(cell->Index());

        // Transform UVs to image coordinates
        std::transform(
            uvs.begin(), uvs.end(), uvs.begin(),
            [maxXIdx, maxYIdx](const cv::Vec2d& p) -> cv::Vec2d {
                return {p[0] * maxXIdx, p[1] * maxYIdx};
            });

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

auto ComputeOBB(vtkPolyData* mesh)
{
    struct obb_result {
        cv::Vec3d origin{0, 0, 0};
        cv::Vec3d xAxis{1, 0, 0};
        cv::Vec3d yAxis{0, 1, 0};
        cv::Vec3d zAxis{0, 0, 1};
        std::array<double, 3> size{1, 1, 1};
    } res;

    const auto obbTree = vtkSmartPointer<vtkOBBTree>::New();
    obbTree->ComputeOBB(
        mesh, res.origin.val, res.xAxis.val, res.yAxis.val, res.zAxis.val,
        res.size.data());

    return res;
}

auto AlignVectorToVector(cv::Vec3d a, const cv::Vec3d& b, const cv::Vec3d& c) -> cv::Mat
{
    // Identity initial rotation
    cv::Mat r = cv::Mat::eye(3, 3, CV_64F);

    // First, check for anti-parallel vectors at low precision
    if (almost_equal(a.dot(b), -1., 1e-3)) {
        auto d = abs(c) - cv::Vec3d{1., 1., 1.};
        d = copysign(cv::Vec3d{1., 1., 1.}, d);
        r.diag() = d;
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
    const cv::Vec3d& y) -> UVMap
{
    UVMap out;

    const auto uLen = cv::norm(x);
    const auto vLen = cv::norm(y);
    const auto uVec = x / uLen;
    const auto vVec = y / vLen;

    // Add points
    cv::Vec3d p;
    for (const auto ptID : range(mesh->GetNumberOfPoints())) {
        mesh->GetPoint(ptID, p.val);

        auto u = (p - o).dot(uVec) / uLen;
        auto v = (p - o).dot(vVec) / vLen;

        out.addUV({u, v});
    }

    // Add faces
    const auto ptIDs = vtkSmartPointer<vtkIdList>::New();
    UVMap::Face f;
    for (const auto cellIdx : range(mesh->GetNumberOfCells())) {
        mesh->GetCellPoints(cellIdx, ptIDs);
        int idx{0};
        for (const auto ptID : *ptIDs) {
            f[idx++] = ptID;
        }
        out.addFace(cellIdx, f);
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

}  // namespace

void ReorderUnorganizedTexture::setMesh(const ITKMesh::Pointer& mesh)
{
    inputMesh_ = mesh;
}

void ReorderUnorganizedTexture::setUVMap(const UVMap& uv) { inputUV_ = uv; }

void ReorderUnorganizedTexture::setTextureMat(const cv::Mat& img)
{
    inputTexture_ = img;
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

auto ReorderUnorganizedTexture::getUVMap() -> UVMap { return outputUV_; }

auto ReorderUnorganizedTexture::getTextureMat() -> cv::Mat
{
    return outputTexture_;
}

auto ReorderUnorganizedTexture::getDepthMap() -> cv::Mat
{
    return outputDepthMap_;
}

// Compute the result
auto ReorderUnorganizedTexture::compute() -> cv::Mat
{
    create_texture_();
    return outputTexture_;
}

void ReorderUnorganizedTexture::create_texture_()
{
    // Compute mesh's (rough) OBB
    auto mesh = rt::ITK2VTK(inputMesh_);
    auto [origin, xAxis, yAxis, zAxis, size] = ComputeOBB(mesh);

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
    std::vector<Triangle> tris;
    auto ptIDs = vtkSmartPointer<vtkIdList>::New();
    Vector3 a, b, c;
    for (auto cellIdx : range(mesh->GetNumberOfCells())) {
        mesh->GetCellPoints(cellIdx, ptIDs);

        mesh->GetPoint(ptIDs->GetId(0), a.values);
        mesh->GetPoint(ptIDs->GetId(1), b.values);
        mesh->GetPoint(ptIDs->GetId(2), c.values);

        // Add the face to the BVH tree
        tris.emplace_back(a, b, c);
    }

    bvh::v2::ThreadPool threadPool;
    bvh::v2::ParallelExecutor executor(threadPool);

    // Get triangle centers and bounding boxes (required for BVH builder)
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
    auto bvh = bvh::v2::DefaultBuilder<Node>::build(
        threadPool, bboxes, centers, config);

    // This precomputes some data to speed up traversal further
    std::vector<PrecomputedTri> precompTris(tris.size());
    executor.for_each(0, tris.size(), [&](const auto begin, const auto end) {
        for (auto i = begin; i < end; ++i) {
            auto j = bvh.prim_ids[i];
            precompTris[i] = tris[j];
        }
    });

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
            sampleRate = ::ComputeUVDensity(
                inputMesh_, inputUV_, inputTexture_.cols, inputTexture_.rows);
            cols = static_cast<int>(std::ceil(xLen / sampleRate));
            rows = static_cast<int>(std::ceil(yLen / sampleRate));
            break;
    }

    logger()->debug(
        "Output size: {}x{} (Sample rate: {:.5g})", cols, rows, sampleRate);

    // Set up the output image
    outputTexture_ = cv::Mat::zeros(rows, cols, CV_8UC3);
    outputDepthMap_ = cv::Mat::zeros(rows, cols, CV_32FC1);

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

    for (auto [v, u] : range2D(rows, cols)) {
        // Convert pixel position to offset in mesh's XY space
        auto uOffset = u * sampleRate * normedX;
        auto vOffset = v * sampleRate * normedY;

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

        // Assign distance to depth map
        outputDepthMap_.at<float>(v, u) = static_cast<float>(hit.value().distance);

        // Cell info
        auto cellId = bvh.prim_ids[hit.value().primitiveIdx];

        // Get the 2D and 3D pts
        std::vector<cv::Vec3d> uvPts;
        for (const auto& uv : inputUV_.getFaceUVs(cellId)) {
            uvPts.emplace_back(uv[0], uv[1], 0.0);
        }

        // Intersection point
        auto inter = hit.value().intersection;
        cv::Vec3d bCoord{inter.u, inter.v, 1 - inter.u - inter.v};

        // Get the UV position of the intersection point
        // Inexplicably, bvh barycentric coordinates are relative to the 2nd
        // pt?
        auto cPoint = ::BaryToXYZ(bCoord, uvPts[1], uvPts[2], uvPts[0]);

        // Convert the UV position to pixel coordinates (in orig image)
        auto x = static_cast<float>(cPoint[0] * (inputTexture_.cols - 1));
        auto y = static_cast<float>(cPoint[1] * (inputTexture_.rows - 1));

        // Bilinear interpolate color and assign to output
        cv::Mat subRect;
        cv::getRectSubPix(inputTexture_, {1, 1}, {x, y}, subRect);
        outputTexture_.at<cv::Vec3b>(v, u) = subRect.at<cv::Vec3b>(0, 0);
    }

    outputUV_ = CreateUVMap(mesh, origin, xAxis, yAxis);
}
