#include <iostream>

#include <boost/program_options.hpp>
#include <educelab/core/types/Mesh.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "rt/filesystem.hpp"
#include "rt/io/ImageIO.hpp"
#include "rt/io/MeshIO.hpp"
#include "rt/types/Mesh.hpp"

namespace fs = rt::filesystem;
namespace po = boost::program_options;
namespace cvl = cv::utils::logging;

auto main(int argc, char** argv) -> int
{
    ///// Parse the command line options /////
    // All command line options
    // clang-format off
    po::options_description required("General Options");
    required.add_options()
        ("help,h", "Show this message")
        ("input-mesh,i", po::value<std::string>()->required(), "Input mesh file")
        ("texture,t", po::value<std::string>()->required(), "New texture image")
        ("compute-normals", po::bool_switch(), "Compute mesh normals")
        ("output-mesh,o", po::value<std::string>()->required(), "Output mesh file");

    po::options_description all("Usage");
    all.add(required);
    // clang-format on

    // Parse the cmd line
    po::variables_map parsed;
    po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);

    // Show the help message
    if (parsed.count("help") > 0 || argc < 2) {
        std::cout << all << std::endl;
        return EXIT_SUCCESS;
    }

    // Warn of missing options
    try {
        po::notify(parsed);
    } catch (po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Silence OpenCV logging
    cvl::setLogLevel(cvl::LogLevel::LOG_LEVEL_SILENT);

    // Load mesh
    const fs::path inputPath = parsed["input-mesh"].as<std::string>();
    auto reader = rt::io::ReadMesh(inputPath);

    // Load the image
    const fs::path imagePath = parsed["texture"].as<std::string>();

    // Compute normals (angle-weighted; see ADR 0001)
    auto mesh = reader.mesh;
    if (parsed["compute-normals"].as<bool>()) {
        for (std::size_t vid = 0; vid < mesh->num_vertices(); ++vid) {
            mesh->vertex(vid).normal = educelab::vertex_normal(*mesh, vid);
        }
    }

    // Write the new mesh
    const fs::path outputPath = parsed["output-mesh"].as<std::string>();
    rt::io::WriteMesh(outputPath, *mesh, reader.uvMap, imagePath);

    return EXIT_SUCCESS;
}
