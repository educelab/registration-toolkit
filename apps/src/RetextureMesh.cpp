#include <iostream>

#include <boost/program_options.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "rt/filesystem.hpp"
#include "rt/io/ImageIO.hpp"
#include "rt/io/OBJReader.hpp"
#include "rt/io/OBJWriter.hpp"
#include "rt/util/CalculateNormals.hpp"

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
    rt::io::OBJReader reader;
    reader.setPath(inputPath);
    reader.read();

    // Load the image
    const fs::path imagePath = parsed["texture"].as<std::string>();

    // Compute normals
    auto mesh = reader.getMesh();
    if (parsed["compute-normals"].as<bool>()) {
        rt::CalculateNormals calcNormals(mesh);
        mesh = calcNormals.compute();
    }

    // Write the new mesh
    const fs::path outputPath = parsed["output-mesh"].as<std::string>();
    rt::io::OBJWriter writer;
    writer.setPath(outputPath);
    writer.setMesh(mesh);
    writer.setUVMap(reader.getUVMap());
    writer.setTextureSource(imagePath);
    writer.write();

    return EXIT_SUCCESS;
}
