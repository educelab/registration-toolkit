#include <string>

#include <OpenABF/OpenABF.hpp>
#include <boost/program_options.hpp>

#include "rt/Logging.hpp"
#include "rt/ReorderUnorganizedTexture.hpp"
#include "rt/filesystem.hpp"
#include "rt/io/FileExtensionFilter.hpp"
#include "rt/io/ImageIO.hpp"
#include "rt/io/OBJReader.hpp"
#include "rt/io/OBJWriter.hpp"
#include "rt/types/ITKMesh.hpp"

namespace po = boost::program_options;
namespace fs = rt::filesystem;
namespace abf = OpenABF;
using namespace rt;

using ABF = abf::ABFPlusPlus<double>;
using LSCM = abf::AngleBasedLSCM<double, ABF::Mesh>;
using Mesh = ABF::Mesh;

namespace
{
auto ITKtoABF(const ITKMesh::Pointer& mesh) -> Mesh::Pointer
{
    auto res = Mesh::New();
    logger()->debug("[ITKtoABF] Copying vertices");
    for (auto pt = mesh->GetPoints()->Begin(); pt != mesh->GetPoints()->End();
         ++pt) {
        res->insert_vertex(pt->Value());
    }

    logger()->debug("[ITKtoABF] Copying faces");
    for (const auto cell : *mesh->GetCells()) {
        res->insert_face(cell->GetPointIdsContainer());
    }
    res->update_boundary();
    return res;
}

auto ABFtoITK(const Mesh::Pointer& mesh) -> ITKMesh::Pointer
{
    auto res = ITKMesh::New();

    logger()->debug("[ABFtoITK] Copying vertices");
    for (const auto& v : mesh->vertices()) {
        res->SetPoint(v->idx, v->pos.data());
        res->SetPointData(v->idx, v->normal().data());
    }

    ITKCell::CellAutoPointer cell;
    logger()->debug("[ABFtoITK] Copying faces");
    for (const auto& f : mesh->faces()) {
        int idx{0};
        cell.TakeOwnership(new ITKTriangle);
        for (const auto& e : *f) {
            cell->SetPointId(idx++, e->vertex->idx);
        }
        res->SetCell(f->idx, cell);
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
    io::OBJReader reader;
    reader.setPath(inPath);
    const auto in = reader.read();

    logger()->debug("Converting to HEM");
    auto hem = ITKtoABF(in);

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

    logger()->debug("Converting back to ITK mesh");
    const auto flat = ABFtoITK(hem);

    logger()->info("Reordering texture");
    ReorderUnorganizedTexture reorder;
    reorder.setMesh(flat);
    reorder.setUVMap(reader.getUVMap());
    reorder.setTextureMat(reader.getTextureMat());
    reorder.setSamplingMode(ReorderUnorganizedTexture::SamplingMode::OutputWidth);
    reorder.setSampleDim(8192);
    const auto texture = reorder.compute();

    if (FileExtensionFilter(outPath, {"jpg", "jpeg", "png", "tiff", "tif"})) {
        logger()->info("Writing image: {}", outPath.string());
        WriteImage(outPath, texture);
    } else if (FileExtensionFilter(outPath, {"obj"})) {
        logger()->info("Writing mesh: {}", outPath.string());
        io::OBJWriter writer;
        writer.setPath(outPath);
        writer.setMesh(in);
        writer.setUVMap(reorder.getUVMap());
        writer.setTexture(texture);
        writer.write();
    } else {
        logger()->error("Unsupported output format: {}", outPath.string());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
