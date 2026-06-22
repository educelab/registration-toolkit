#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include <boost/program_options.hpp>
#include <educelab/core/utils/String.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <smgl/Graphviz.hpp>
#include <smgl/smgl.hpp>

#include "rt/Logging.hpp"
#include "rt/ReorderUnorganizedTexture.hpp"
#include "rt/Version.hpp"
#include "rt/filesystem.hpp"
#include "rt/graph.hpp"
#include "rt/io/FileExtensionFilter.hpp"

namespace fs = rt::filesystem;
namespace po = boost::program_options;
namespace cvl = cv::utils::logging;

using namespace rt;
using namespace rt::graph;
using namespace educelab;

using SamplingOrigin = ReorderUnorganizedTexture::SamplingOrigin;
std::unordered_map<std::string, SamplingOrigin> StrToOrigin{
    {"tl", SamplingOrigin::TopLeft},
    {"tr", SamplingOrigin::TopRight},
    {"bl", SamplingOrigin::BottomLeft},
    {"br", SamplingOrigin::BottomRight},
};

using SamplingMode = ReorderUnorganizedTexture::SamplingMode;
std::unordered_map<std::string, SamplingMode> StrToMode{
    {"rate", SamplingMode::Rate},
    {"width", SamplingMode::OutputWidth},
    {"height", SamplingMode::OutputHeight},
    {"auto", SamplingMode::AutoUV},
};

using ProjectionParams = ReorderUnorganizedTexture::ProjectionParams;

// Parse a plain-text pinhole camera description (see the --camera-file help)
// into ProjectionParams. Returns std::nullopt and logs the reason if the file
// can't be opened or is missing/malformed entries.
static auto ParseCameraFile(const fs::path& path)
    -> std::optional<ProjectionParams>
{
    std::ifstream camFile(path);
    if (not camFile) {
        rt::logger()->error("Could not open camera file: {}", path.string());
        return std::nullopt;
    }

    // Collect tokens, stripping '#' comments line-by-line
    std::stringstream tokens;
    std::string line;
    while (std::getline(camFile, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line.erase(hash);
        }
        tokens << line << ' ';
    }

    ProjectionParams p;
    bool haveFx{false}, haveFy{false}, haveCx{false}, haveCy{false};
    bool haveW{false}, haveH{false}, havePose{false};
    std::string key;
    auto readScalar = [&](auto& out, const char* name) -> bool {
        if (not(tokens >> out)) {
            rt::logger()->error(
                "Camera file: missing/invalid value for '{}'", name);
            return false;
        }
        return true;
    };
    while (tokens >> key) {
        key = to_lower_copy(key);
        if (key == "fx") {
            if (not readScalar(p.fx, "fx")) return std::nullopt;
            haveFx = true;
        } else if (key == "fy") {
            if (not readScalar(p.fy, "fy")) return std::nullopt;
            haveFy = true;
        } else if (key == "cx") {
            if (not readScalar(p.cx, "cx")) return std::nullopt;
            haveCx = true;
        } else if (key == "cy") {
            if (not readScalar(p.cy, "cy")) return std::nullopt;
            haveCy = true;
        } else if (key == "width") {
            if (not readScalar(p.width, "width")) return std::nullopt;
            haveW = true;
        } else if (key == "height") {
            if (not readScalar(p.height, "height")) return std::nullopt;
            haveH = true;
        } else if (key == "pose") {
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    if (not(tokens >> p.extrinsics(r, c))) {
                        rt::logger()->error(
                            "Camera file: 'pose' needs 16 numeric values "
                            "(row-major 4x4)");
                        return std::nullopt;
                    }
                }
            }
            havePose = true;
        } else {
            rt::logger()->error("Camera file: unknown key '{}'", key);
            return std::nullopt;
        }
    }

    if (not(haveFx and haveFy and haveCx and haveCy and haveW and haveH and
            havePose)) {
        rt::logger()->error(
            "Camera file must define fx, fy, cx, cy, width, height, and pose");
        return std::nullopt;
    }
    if (const auto err = ValidateProjectionParams(p)) {
        rt::logger()->error("Camera file: {}", *err);
        return std::nullopt;
    }
    return p;
}

auto main(int argc, char* argv[]) -> int
{
    ///// Parse the command line options /////
    // clang-format off
    po::options_description required("General Options");
    required.add_options()
        ("help,h", "Show this message")
        ("input-mesh,i", po::value<std::string>()->required(),
             "Path to input OBJ with unordered texture (i.e. multicharts)")
        ("output-file,o", po::value<std::string>()->required(),
             "Output path. An OBJ extension writes the mesh with its ordered "
             "texture; an image extension (jpg, png, tif) writes just the "
             "ordered texture image.")
        ("depth-map", po::value<std::string>(), "Path to output depth map image")
        ("position-map", po::value<std::string>(),
             "Path to output 3D position map image (CV_32FC3; per-pixel XYZ)")
        ("sampling-origin", po::value<std::string>()->default_value("tl"),
             "Origins: tl, tr, bl, br")
        ("sampling-mode,m", po::value<std::string>()->default_value("auto"),
             "Modes: rate, width, height, auto. If 'rate', specify "
             "the sampling step size in mesh units using --sampling-rate. If "
             "'width' or 'height', specify the length of the corresponding "
             "dimension in pixels using --sampling-dim. If 'auto', the sample "
             "automatically calculated from the average pixel density of the "
             "input mesh.")
        ("sampling-rate,r", po::value<double>()->default_value(0.1),
             "If --sampling-mode is 'rate', the pixel size in mesh units")
        ("sampling-dim,d", po::value<std::size_t>()->default_value(800),
             "If --sampling-mode is 'width' or 'height', the length of the "
             "corresponding output dimension in pixels")
        ("use-first-intersection,f", "This program assumes that "
             "the projection origin is behind the base plane of the sampled "
             "mesh. Thus, the last mesh intersection point will lie on the "
             "visible surface. If instead the projection origin is in front of "
             "the base plane, the first mesh intersection point lies on the "
             "visible surface.");

    po::options_description projOptions("Projection Options");
    projOptions.add_options()
        ("projection", po::value<std::string>()->default_value("orthographic"),
             "Projection model: orthographic (default) or camera. 'camera' "
             "renders the textured mesh through a pinhole camera; if "
             "--camera-file is omitted, a camera is auto-derived to frame the "
             "mesh.")
        ("camera-file", po::value<std::string>(),
             "Path to a plain-text file describing the pinhole camera "
             "intrinsics and world-to-camera pose. The file is a set of "
             "whitespace-separated key/value entries (order-independent; '#' "
             "starts a comment):\n"
             "  fx <px>\n  fy <px>\n  cx <px>\n  cy <px>\n"
             "  width <px>\n  height <px>\n"
             "  pose <16 values>\n"
             "'pose' is the world-to-camera 4x4 matrix in row-major order "
             "(OpenCV convention x_cam = R*X + t); its 16 values may span "
             "multiple lines.");

    po::options_description graphOptions("Render Graph Options");
    graphOptions.add_options()
    ("output-graph,g", po::value<std::string>(), "Render graph JSON file")
    ("output-dot", po::value<std::string>(), "Render graph Dot file");

    po::options_description all("Usage");
    all.add(required).add(projOptions).add(graphOptions);
    // clang-format on

    // Parse the cmd line
    po::variables_map parsed;
    po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);

    // Show the help message
    if (parsed.count("help") > 0 || argc < 5) {
        std::cerr << all << std::endl;
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

    fs::path inputPath = parsed["input-mesh"].as<std::string>();
    fs::path outputPath = parsed["output-file"].as<std::string>();

    // Get parameters
    auto originStr = to_lower_copy(parsed["sampling-origin"].as<std::string>());
    auto samplingOrigin = StrToOrigin.at(originStr);
    auto modeStr = to_lower_copy(parsed["sampling-mode"].as<std::string>());
    auto sampleMode = StrToMode.at(modeStr);
    auto sampleRate = parsed["sampling-rate"].as<double>();
    auto sampleDim = parsed["sampling-dim"].as<std::size_t>();
    auto useFirstIntersection = parsed.count("use-first-intersection") > 0;

    // Resolve the projection model
    using ProjectionMode = ReorderUnorganizedTexture::ProjectionMode;
    auto projStr = to_lower_copy(parsed["projection"].as<std::string>());
    if (projStr != "orthographic" and projStr != "camera") {
        rt::logger()->error("Unknown projection model: {}", projStr);
        return EXIT_FAILURE;
    }
    auto projectionMode = (projStr == "camera") ? ProjectionMode::Camera
                                                : ProjectionMode::Orthographic;

    // Parse an explicit camera, if one was given
    std::optional<ProjectionParams> projParams;
    if (projStr == "camera") {
        if (parsed.count("camera-file") > 0) {
            projParams = ParseCameraFile(parsed["camera-file"].as<std::string>());
            if (not projParams) {
                return EXIT_FAILURE;
            }
        } else {
            rt::logger()->info(
                "No explicit camera given; auto-deriving camera from mesh");
        }
    }

    ///// Start render graph /////
    rt::graph::RegisterNodes();
    smgl::Graph graph;

    // Add the project metadata
    graph.setProjectMetadata({{ProjectInfo::Name(), ProjectMetadata()}});

    ///// Setup caching /////
    if (parsed.count("output-graph") > 0) {
        fs::path cacheFile = parsed["output-graph"].as<std::string>();
        graph.setEnableCache(true);
        graph.setCacheFile(cacheFile);
    }

    // Load the mesh
    auto reader = graph.insertNode<MeshReadNode>();
    reader->path = inputPath;

    // We don't support RGBA textures
    auto convert = graph.insertNode<ColorConvertNode>();
    convert->imageIn = reader->image;
    convert->channels = 3;

    // Reorder the texture
    auto reorder = graph.insertNode<ReorderTextureNode>();
    reorder->meshIn = reader->mesh;
    reorder->uvMapIn = reader->uvMap;
    reorder->imageIn = convert->imageOut;
    reorder->samplingOrigin = samplingOrigin;
    reorder->samplingMode = sampleMode;
    reorder->sampleRate = sampleRate;
    reorder->sampleDim = sampleDim;
    reorder->useFirstIntersection = useFirstIntersection;
    reorder->projectionMode = projectionMode;
    if (projParams) {
        reorder->projectionParams = *projParams;
    }

    // Write to file: an image-format output gets just the reordered texture
    // image; any other extension is treated as a textured mesh.
    if (FileExtensionFilter(outputPath, {"jpg", "jpeg", "png", "tiff", "tif"})) {
        auto writer = graph.insertNode<WriteImageNode>();
        writer->path = outputPath;
        writer->image = reorder->imageOut;
    } else {
        auto writer = graph.insertNode<MeshWriteNode>();
        writer->path = outputPath;
        writer->mesh = reader->mesh;
        writer->uvMap = reorder->uvMapOut;
        writer->image = reorder->imageOut;
    }

    // Write depth map
    if (parsed.count("depth-map") > 0) {
        auto imgWriter = graph.insertNode<WriteImageNode>();
        imgWriter->path = parsed["depth-map"].as<std::string>();
        imgWriter->image = reorder->depthMapOut;
    }

    // Write 3D position map
    if (parsed.count("position-map") > 0) {
        auto posWriter = graph.insertNode<WriteImageNode>();
        posWriter->path = parsed["position-map"].as<std::string>();
        posWriter->image = reorder->positionMapOut;
    }

    // Compute result
    graph.update();

    // Write Dot file
    if (parsed.count("output-dot") > 0) {
        smgl::WriteDotFile(parsed["output-dot"].as<std::string>(), graph);
    }
}
