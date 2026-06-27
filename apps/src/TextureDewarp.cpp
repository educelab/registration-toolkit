#include <string>

#include <OpenABF/OpenABF.hpp>
#include <boost/program_options.hpp>
#include <educelab/core/utils/Filesystem.hpp>

#include "rt/Logging.hpp"
#include "rt/ReorderUnorganizedTexture.hpp"
#include "rt/filesystem.hpp"
#include "rt/io/ImageIO.hpp"
#include "rt/io/MeshIO.hpp"
#include "rt/types/Mesh.hpp"

namespace po = boost::program_options;
namespace fs = rt::filesystem;
namespace abf = OpenABF;
using namespace rt;

using ABF = abf::ABFPlusPlus<double>;
using LSCM = abf::AngleBasedLSCM<double, ABF::Mesh>;
using ABFMesh = ABF::Mesh;

namespace
{
auto MeshToABF(const rt::Mesh::Pointer& mesh) -> ABFMesh::Pointer
{
    auto res = ABFMesh::New();
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

auto ABFToMesh(const ABFMesh::Pointer& mesh) -> rt::Mesh::Pointer
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
        ("log-level", po::value<std::string>()->default_value("info"),
            "Log level: debug, info, warning, error, critical, off");
    // clang-format on

    po::variables_map args;
    po::store(po::command_line_parser(argc, argv).options(required).run(), args);

    if (args.count("help") > 0 or argc < 2) {
        std::cerr << required << std::endl;
        return EXIT_SUCCESS;
    }

    try {
        po::notify(args);
    } catch (po::error& e) {
        rt::logger()->error("{}", e.what());
        return EXIT_FAILURE;
    }

    rt::set_log_level(args["log-level"].as<std::string>());

    const fs::path inPath = args["input"].as<std::string>();
    const fs::path outPath = args["output"].as<std::string>();

    logger()->info("Loading mesh: {}", inPath.string());
    auto reader = rt::ReadMesh(inPath);
    const auto in = reader.mesh;

    logger()->debug("Converting to HEM");
    auto hem = MeshToABF(in);

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

    logger()->debug("Solving LSCM");
    LSCM::Compute(hem);

    logger()->debug("Converting back to canonical mesh");
    const auto flat = ABFToMesh(hem);

    logger()->info("Reordering texture");
    ReorderUnorganizedTexture reorder;
    reorder.setMesh(flat);
    reorder.setUVMap(reader.uvMap);
    reorder.setTextureMat(reader.texture);
    reorder.setSamplingMode(ReorderUnorganizedTexture::SamplingMode::AutoUV);
    const auto texture = reorder.compute();

    if (educelab::is_file_type(outPath, "jpg", "jpeg", "png", "tiff", "tif")) {
        logger()->info("Writing image: {}", outPath.string());
        WriteImage(outPath, texture);
    } else if (educelab::is_file_type(outPath, "obj")) {
        logger()->info("Writing mesh: {}", outPath.string());
        rt::WriteMesh(outPath, *in, reorder.getUVMap(), texture);
    } else {
        logger()->error("Unsupported output format: {}", outPath.string());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
