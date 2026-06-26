#include <limits>
#include <stdexcept>
#include <string>

#include <OpenABF/OpenABF.hpp>
#include <boost/program_options.hpp>
#include <educelab/core/types/Mat.hpp>
#include <educelab/core/utils/Iteration.hpp>
#include <educelab/core/utils/String.hpp>
#include <opencv2/core.hpp>

#include <educelab/core/utils/Filesystem.hpp>

#include "rt/Logging.hpp"
#include "rt/ReorderUnorganizedTexture.hpp"
#include "rt/filesystem.hpp"
#include "rt/io/ImageIO.hpp"
#include "rt/io/MeshIO.hpp"
#include "rt/types/Mesh.hpp"

namespace po = boost::program_options;
namespace fs = rt::filesystem;
namespace el = educelab;
namespace abf = OpenABF;

using ABF = abf::ABFPlusPlus<double>;
using LSCM = abf::AngleBasedLSCM<double, ABF::Mesh>;
using AbfMesh = ABF::Mesh;
using Mat = el::Mat<4, 4, double>;
using namespace rt;

namespace
{
auto MeshToABF(const rt::Mesh::Pointer& mesh) -> AbfMesh::Pointer
{
    auto res = AbfMesh::New();
    logger()->debug("[MeshToABF] Copying vertices");
    for (std::size_t vid = 0; vid < mesh->num_vertices(); ++vid) {
        const auto& v = mesh->vertex(vid);
        res->insert_vertex(v[0], v[1], v[2]);
    }

    logger()->debug("[MeshToABF] Copying faces");
    for (std::size_t fid = 0; fid < mesh->num_faces(); ++fid) {
        res->insert_face(mesh->face(fid));
    }
    res->update_boundary();
    return res;
}

auto ABFToMesh(const AbfMesh::Pointer& mesh) -> rt::Mesh::Pointer
{
    auto res = rt::Mesh::New();

    logger()->debug("[ABFToMesh] Copying vertices");
    for (const auto& v : mesh->vertices()) {
        const auto idx = res->insert_vertex(v->pos[0], v->pos[1], v->pos[2]);
        const auto n = v->normal();
        res->vertex(idx).normal = educelab::Vec<double, 3>{n[0], n[1], n[2]};
    }

    logger()->debug("[ABFToMesh] Copying faces");
    for (const auto& f : mesh->faces()) {
        rt::Mesh::Face face;
        for (const auto& e : *f) {
            face.push_back(e->vertex->idx);
        }
        res->insert_face(face);
    }

    return res;
}

auto GetAABB(const AbfMesh::Pointer& mesh) -> std::pair<abf::Vec3d, abf::Vec3d>
{
    abf::Vec3d min, max;
    min.fill(abf::INF<double>);
    max.fill(-abf::INF<double>);
    for (std::size_t vid = 0; vid < mesh->num_vertices(); ++vid) {
        const auto& p = mesh->vertex(vid)->pos;
        for (std::size_t c = 0; c < 3; ++c) {
            min[c] = std::min(p[c], min[c]);
            max[c] = std::max(p[c], max[c]);
        }
    }
    return {min, max};
}

enum class Axis { X, Y, Z };

auto GetArea(const AbfMesh::Pointer& mesh, Axis axis) -> double
{
    auto [min, max] = GetAABB(mesh);
    if (axis == Axis::X) {
        return (max[1] - min[1]) * (max[2] - min[2]);
    }
    if (axis == Axis::Y) {
        return (max[0] - min[0]) * (max[2] - min[2]);
    }

    return (max[0] - min[0]) * (max[1] - min[1]);
}

template <typename T1, typename T2>
auto matmul(const el::Mat<4, 4, T1>& mat, const abf::Vec<T2, 3>& vec)
    -> abf::Vec<T2, 3>
{
    abf::Vec3d res;
    for (std::size_t m{0}; m < 3; m++) {
        res[m] = mat(m, 3);
        for (std::size_t n{0}; n < 3; n++) {
            res[m] += mat(m, n) * vec[n];
        }
    }
    return res;
}

auto Rotate4x4(const double radians, abf::Vec3d vec) -> Mat
{
    vec = vec.unit();
    const auto x = vec[0];
    const auto y = vec[1];
    const auto z = vec[2];
    const auto s = std::sin(radians);
    const auto c = std::cos(radians);
    return Mat{
        x * x * (1 - c) + c,
        x * y * (1 - c) - z * s,
        x * z * (1 - c) + y * s,
        0,
        y * x * (1 - c) + z * s,
        y * y * (1 - c) + c,
        y * z * (1 - c) - x * s,
        0,
        x * z * (1 - c) - y * s,
        y * z * (1 - c) + x * s,
        z * z * (1 - c) + c,
        0,
        0,
        0,
        0,
        1};
}

void ApplyTransform(AbfMesh::Pointer mesh, const Mat& tfm)
{
    for (std::size_t vid = 0; vid < mesh->num_vertices(); ++vid) {
        mesh->vertex(vid)->pos = matmul(tfm, mesh->vertex(vid)->pos);
    }
}

void MinimizeBBox(const AbfMesh::Pointer& mesh, const Axis axis)
{
    abf::Vec3d vec;
    if (axis == Axis::X) {
        vec = {1, 0, 0};
    } else if (axis == Axis::Y) {
        vec = {0, 1, 0};
    } else {
        vec = {0, 0, 1};
    }

    constexpr double start{-45.};
    constexpr double end{45.};
    constexpr double step{0.5};
    const auto tmp = mesh->clone();
    const auto initialTfm = Rotate4x4(el::to_radians<double>(start), vec);
    const auto stepTfm = Rotate4x4(el::to_radians<double>(step), vec);
    ApplyTransform(tmp, initialTfm);
    double minAngle{0};
    double minArea{abf::INF<double>};
    for (const auto angle : el::range(start, end, step)) {
        if (const auto area = GetArea(tmp, axis); area < minArea) {
            minAngle = angle;
            minArea = area;
        }
        ApplyTransform(tmp, stepTfm);
    }

    // Return final result
    logger()->debug("Minimum angle: {}", minAngle);
    const auto finalTfm = Rotate4x4(el::to_radians<double>(minAngle), vec);
    ApplyTransform(mesh, finalTfm);
}

auto GetEndpoints(AbfMesh::Pointer& mesh) -> std::pair<std::size_t, std::size_t>
{
    // Create a mat of 3D points
    const auto nVerts = static_cast<int>(mesh->num_vertices());
    cv::Mat verts = cv::Mat::zeros(nVerts, 3, CV_64FC1);
    for (int vid = 0; vid < nVerts; ++vid) {
        const auto p = mesh->vertex(vid)->pos;
        for (int c = 0; c < 3; ++c) {
            verts.at<double>(vid, c) = p[c];
        }
    }

    // const cv::PCA pca(verts, cv::Mat(), cv::PCA::DATA_AS_ROW, 3);
    // verts = pca.project(verts);
    // for (int vid = 0; vid < nVerts; ++vid) {
    //     for (int c = 0; c < 3; ++c) {
    //         mesh->vertex(vid)->pos[c] = verts.at<double>(vid, c);
    //     }
    // }

    // Center on the shell barycenter
    abf::Vec3d barycenter{0, 0, 0};
    double totalArea = 0.;
    for (std::size_t fid = 0; fid < mesh->num_faces(); ++fid) {
        const auto f = mesh->face(fid);
        const auto area = f->area();
        barycenter += f->barycenter() * area;
        totalArea += area;
    }
    barycenter /= totalArea;
    for (std::size_t vid = 0; vid < mesh->num_vertices(); ++vid) {
        mesh->vertex(vid)->pos -= barycenter;
    }

    logger()->debug("Minimize around Y axis");
    MinimizeBBox(mesh, Axis::Y);
    logger()->debug("Minimize around Z axis");
    MinimizeBBox(mesh, Axis::Z);

    // Get the mesh boundaries
    const auto boundaries = mesh->boundaries();
    if (boundaries.size() < 2) {
        throw std::runtime_error(
            "Expected 2 boundaries but found " +
            std::to_string(boundaries.size()) +
            ", cannot find seam endpoints");
    }
    if (boundaries.size() > 2) {
        logger()->warn(
            "Expected 2 boundaries but found {}, seam may be incorrect",
            boundaries.size());
    }

    // Find the best start and end points
    auto [min, max] = GetAABB(mesh);
    constexpr auto kInvalidIdx = std::numeric_limits<std::size_t>::max();
    std::size_t start{kInvalidIdx};
    std::size_t end{kInvalidIdx};
    double startDist = abf::INF<double>;
    abf::Vec3d ref{max[0], 0., max[2]};
    for (const auto& e : boundaries[0]) {
        const auto dist = (e->vertex->pos - ref).magnitude();
        if (dist < startDist) {
            start = e->vertex->idx;
            startDist = dist;
        }
    }

    double endDist = abf::INF<double>;
    ref[0] = min[0];
    for (const auto& e : boundaries[1]) {
        const auto dist = (e->vertex->pos - ref).magnitude();
        if (dist < endDist) {
            end = e->vertex->idx;
            endDist = dist;
        }
    }

    if (start == kInvalidIdx or end == kInvalidIdx) {
        throw std::runtime_error("Failed to find seam endpoints on boundaries");
    }

    return {start, end};
}

enum class MeshType { Original, Centered, Flattened };

}  // namespace

auto main(int argc, const char* argv[]) -> int
{

    ///// Parse the command line options /////
    // clang-format off
    po::options_description required("General Options");
    required.add_options()
        ("help,h", "Show this message")
        ("input,i", po::value<std::string>()->required(), "Input mesh")
        ("output,o", po::value<std::string>()->required(), "Output file path (mesh or image)")
        ("mesh-type,m", po::value<std::string>()->default_value("centered"), "The output mesh used when saving an OBJ output: original, centered, flat")
        ("log-level", po::value<std::string>()->default_value("info"),
            "Log level: debug, info, warning, error, critical, off");

    // Parse the cmd line
    po::variables_map args;
    po::store(po::command_line_parser(argc, argv).options(required).run(), args);

    // Show the help message
    if (args.count("help") > 0 or argc < 2) {
        std::cerr << required << std::endl;
        return EXIT_SUCCESS;
    }

    // Warn of missing options
    try {
        po::notify(args);
    } catch (po::error& e) {
        rt::logger()->error("{}", e.what());
        return EXIT_FAILURE;
    }

    // Set log level
    rt::set_log_level(args["log-level"].as<std::string>());

    // Get files
    const fs::path inPath = args["input"].as<std::string>();
    const fs::path outPath = args["output"].as<std::string>();

    // Parse the output mesh type
    const auto typeStr = el::to_lower_copy(args["mesh-type"].as<std::string>());
    auto meshType{MeshType::Original};
    if(typeStr == "original") {
        meshType = MeshType::Original;
    } else if(typeStr == "centered") {
        meshType = MeshType::Centered;
    } else if(typeStr == "flat" or typeStr == "flattened") {
        meshType = MeshType::Flattened;
    } else {
        logger()->error("Unrecognized mesh type: {}", typeStr);
        return EXIT_FAILURE;
    }

    logger()->info("Loading mesh: {}", inPath.string());
    auto reader = rt::io::ReadMesh(inPath);
    const auto in = reader.mesh;

    logger()->debug("Converting to HEM");
    auto hem = MeshToABF(in);
    auto [start, end] = GetEndpoints(hem);
    auto centered = ABFToMesh(hem);

    logger()->info("Finding seam");
    const auto path = abf::FindEdgePath(hem, start, end);
    logger()->debug("Adding seam from v:{} -> v:{}", start, end);
    hem->split_path(path);

    logger()->info("Flattening mesh");
    logger()->debug("Solving ABF++");
    std::size_t iters{0};
    double grad{0};
    try {
        ABF::Compute(hem, iters, grad, 10);
    } catch (const abf::SolverException& e) {
        logger()->warn("Failed to solve ABF++. Falling back to LSCM.");
        logger()->debug("SolverException: {}", e.what());
    }
    logger()->info(
        "ABF++ Iterations: {} || "
        "Final norm: {:.5g}",
        iters, grad);
    // LSCM
    logger()->debug("Solving LSCM");
    LSCM::Compute(hem);

    logger()->debug("Converting back to canonical mesh");
    const auto flat = ABFToMesh(hem);

    // Reorder texture
    logger()->info("Reordering texture");
    ReorderUnorganizedTexture reorder;
    reorder.setMesh(flat);
    reorder.setUVMap(reader.uvMap);
    reorder.setTextureMat(reader.texture);
    reorder.setSamplingMode(ReorderUnorganizedTexture::SamplingMode::AutoUV);
    const auto texture = reorder.compute();

    if (el::is_file_type(outPath, "jpg", "jpeg", "png", "tiff", "tif")) {
        logger()->info("Writing image: {}", outPath.string());
        WriteImage(outPath, texture);

    } else if (el::is_file_type(outPath, "obj")) {
        logger()->info("Writing mesh: {}", outPath.string());
        auto mesh = in;
        if (meshType == MeshType::Centered) {
            mesh = centered;
        } else if (meshType == MeshType::Flattened) {
            mesh = flat;
        }

        rt::io::WriteMesh(outPath, *mesh, reorder.getUVMap(), texture);
    } else {
        logger()->error("Unsupported output format: {}", outPath.string());
    }
}