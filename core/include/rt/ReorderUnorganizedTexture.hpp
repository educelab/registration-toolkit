#pragma once

/** @file */

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "rt/types/Mesh.hpp"
#include "rt/types/UVMap.hpp"

namespace rt
{

/**
 * @brief Reorder an unorganized texture image into an organized texture using
 * 3D mapping information
 *
 * This algorithm reorganizes embedded color information from an input mesh
 * into a new, organized texture image. The effect is similar to taking a
 * snapshot from "above" the input mesh: faces that are adjacent in 3D are also
 * adjacent in the new 2D texture image.
 *
 * This algorithm assumes the input mesh (and therefore the resulting
 * reorganized texture) is roughly planar. The mesh is realigned such that its
 * shortest dimension is perpendicular to the XY plane. The algorithm samples
 * the XY plane into an image, creating one pixel every user-specified interval.
 * A Z-axis aligned ray through the pixel is used to calculate an intersection
 * and correspondence between the mesh and the new image. The color of the
 * intersected point is bilinearly interpolated from the original texture image
 * and placed into the pixel. A UV map is also generated that maps the original
 * input mesh to the new texture image.
 *
 */
class ReorderUnorganizedTexture
{
public:
    /**
     * @brief Bounding box corner to use for the origin of the sampling plane
     */
    enum class SamplingOrigin {
        TopLeft,    /** Set the sampling origin to the top-left corner of the
                       largest face of the bounding box */
        TopRight,   /** Set the sampling origin to the top-right corner of the
                       largest face of the bounding box */
        BottomLeft, /** Set the sampling origin to the bottom-left corner of the
                       largest face of the bounding box */
        BottomRight /** Set the sampling origin to the bottom-right corner of
                       the largest face of the bounding box */
    };

    /** @brief Projection model used to sample the mesh into the output image */
    enum class ProjectionMode {
        Orthographic, /** Parallel rays along the mesh's shortest axis (the
                         default "snapshot from above" behavior) */
        Camera        /** Perspective rays from a pinhole camera center */
    };

    /**
     * @brief How the orientation of the sampling frame is resolved
     *
     * @see setOrientationMode()
     */
    enum class OrientationMode {
        OBB,      /** Take the OBB axis directions as computed. The output
                     image's orientation is arbitrary. (default) */
        Canonical /** Disambiguate the OBB axis directions using the world
                     axes. Requires a canonically oriented input mesh. */
    };

    /**
     * @brief Pinhole camera intrinsics + extrinsics for ProjectionMode::Camera
     *
     * @c extrinsics is the world-to-camera matrix (OpenCV convention:
     * `x_cam = R * X_world + t`, camera looks down +Z, +Y points down). The
     * focal lengths and principal point are in pixels; @c width and @c height
     * are the output image dimensions in pixels. @c k1, @c k2, and @c k3 are
     * radial distortion coefficients (OpenMVG @c pinhole_radial_k3, identical
     * to OpenCV with `p1 = p2 = 0`); all default to zero (no distortion).
     */
    struct ProjectionParams {
        double fx{1.0};
        double fy{1.0};
        double cx{0.0};
        double cy{0.0};
        int width{0};
        int height{0};
        double k1{0.0};
        double k2{0.0};
        double k3{0.0};
        cv::Matx44d extrinsics{cv::Matx44d::eye()};
    };

    /** @brief Sampling rate mode */
    enum class SamplingMode {
        Rate,         /** Use the sample rate provided by setSampleRate() */
        OutputWidth,  /**
                       * Calculate the sample rate needed to produce an output
                       * image with width defined by setSampleDim().
                       */
        OutputHeight, /**
                       * Calculate the sample rate needed to produce an output
                       * image with height defined by setSampleDim().
                       */
        AutoUV        /**
                       * Calculate a sample rate from the average pixel density in the
                       * UV mapped image.
                       */
    };

    /**
     * Default distance (in mesh units) at which to sample the XY plane into
     * image
     */
    constexpr static double DEFAULT_SAMPLE_RATE{0.1};

    /** @brief Set the input mesh */
    void setMesh(const Mesh::Pointer& mesh);
    /** @brief Set the input UV map for the mesh */
    void setUVMap(const UVMap& uv);

    /**
     * @brief Set the input, unorganized texture images
     *
     * The mesh may be textured by more than one image (a multi-chart UV map,
     * e.g. a multi-material OBJ). Images are indexed by UV chart: the color for
     * a face is sampled from `imgs[chart]`, where `chart` is the atlas chart
     * index carried by the face's UV coordinates (see rt::UVMap). A
     * single-texture mesh is simply the one-element case (all faces chart 0).
     *
     * Faces whose chart has no corresponding image (chart index out of range or
     * an empty `cv::Mat`) are left uncolored in the output; compute() emits a
     * single warning naming the affected chart(s).
     *
     * @note Each image is normalized to 8-bit, 3-channel (BGR) on input via
     * rt::QuantizeImage + rt::ColorConvertImage. Higher bit depths and other
     * channel layouts are not yet preserved through the reorder pipeline; see
     * https://github.com/educelab/registration-toolkit/issues/19 for the
     * tracking issue on native multi-bit-depth/channel support.
     */
    void setTextureMats(const std::vector<cv::Mat>& imgs);

    /** @copydoc samplingOrigin() */
    void setSamplingOrigin(SamplingOrigin o);

    /** @brief Sampling bounding box origin */
    [[nodiscard]] auto samplingOrigin() const -> SamplingOrigin;

    /** @copydoc samplingMode() */
    void setSamplingMode(SamplingMode m);

    /** @brief Sampling rate mode */
    [[nodiscard]] auto samplingMode() const -> SamplingMode;

    /**
     * @brief The rate (in mesh units) at which to sample the image plane
     *
     * @see samplingMode()
     */
    void setSampleRate(double s);

    /** @copydoc setSampleRate() */
    [[nodiscard]] auto sampleRate() const -> double;

    /**
     * @brief The size of the output image in pixels
     *
     * Only used if setSamplingMode() is set to SampleMode::OutputWidth or
     * SampleMode::OutputHeight
     *
     * @see samplingMode()
     */
    void setSampleDim(std::size_t d);

    /** @copydoc setSampleDim() */
    [[nodiscard]] auto sampleDim() const -> std::size_t;

    /** @brief Whether to use the first mesh intersection point */
    void setUseFirstIntersection(bool b);

    /** @copydoc setUseFirstIntersection() */
    [[nodiscard]] auto useFirstIntersection() const -> bool;

    /**
     * @brief Set the projection model used to sample the mesh
     *
     * Switching to ProjectionMode::Camera enables perspective sampling. If no
     * projection parameters are provided via setProjectionParams(), a sensible
     * pinhole camera is derived automatically at compute time: positioned along
     * the mesh's shortest OBB axis, looking at the centroid, with intrinsics and
     * output size chosen to frame the mesh (mirroring the orthographic view).
     */
    void setProjectionMode(ProjectionMode m);

    /** @copydoc setProjectionMode() */
    [[nodiscard]] auto projectionMode() const -> ProjectionMode;

    /**
     * @brief Set how the orientation of the sampling frame is resolved
     *
     * The oriented bounding box used to build the sampling frame has arbitrary
     * axis directions, so by default (OrientationMode::OBB) the orientation of
     * the output image is unrelated to the orientation of the input mesh: the
     * texture may come out rotated or mirrored, and successive scans of the
     * same object are not comparable to each other.
     *
     * OrientationMode::Canonical resolves that ambiguity against the world
     * axes, producing an image whose +u runs along the box axis that agrees
     * with world +X and whose +v runs along the one that agrees with world -Y,
     * sampling the surface that faces world +Z. This is only
     * meaningful if the input mesh is already canonically oriented -- mesh
     * right along +X, mesh up along +Y, surface normal along +Z -- as produced
     * by an upstream orientation step; on an arbitrarily posed mesh it merely
     * trades one arbitrary orientation for another.
     *
     * The sampling *plane* is unchanged, so foreshortening behaves exactly as
     * it does under OrientationMode::OBB. The axis *assignment* can change,
     * however: where the world-agreement rule picks different in-plane axes
     * than the extent ordering would, the extent the image width is measured
     * along changes with it, so SamplingMode::OutputWidth and
     * SamplingMode::OutputHeight can yield a different pixel scale than
     * OrientationMode::OBB does.
     *
     * ProjectionMode::Camera honors this as well, by orienting the camera it
     * derives at compute time: the camera is placed on the world +Z side of the
     * surface, with image right along the box axis that agrees with world +X
     * and image down along the one that agrees with world -Y, the same
     * convention the orthographic path produces. That resolves all
     * three choices the raw bounding box leaves arbitrary there -- which side
     * the camera views from, which way is up, and which in-plane axis becomes
     * the image width. A camera supplied through setProjectionParams() is used
     * verbatim and is unaffected.
     *
     * @warning setSamplingOrigin() and setUseFirstIntersection() are both
     * applied downstream of this (orthographic sampling only). Any origin other
     * than SamplingOrigin::TopLeft negates the sampling axes again and undoes
     * the canonical orientation -- SamplingOrigin::TopRight mirrors the result,
     * which is the exact failure this mode exists to prevent. Enabling
     * useFirstIntersection reverses the ray march, so the surface facing world
     * -Z is sampled rather than the +Z-facing one; on a folded or multi-layer
     * fragment that images the wrong layer.
     *
     * @note This resolves the *discrete* ambiguity -- which axis, and which
     * direction along it -- and so removes the 90-degree, 180-degree and
     * mirrored outputs. It deliberately does not reorient the frame onto the
     * world axes: the frame stays the fitted bounding box's own, so the image
     * plane remains parallel to the mesh's base plane rather than to world XY,
     * and a mesh whose normal sits a few degrees off +Z keeps that tilt.
     * Sampling square to the surface is what keeps it from being foreshortened;
     * the world axes only decide which box axis becomes which image axis.
     * ProjectionMode::Camera inherits the same tilt, as a small perspective
     * skew. Nor does this remove the in-plane rotation the bounding-box area
     * minimization applies after realignment (orthographic only), which is
     * bounded by +/-45 degrees and in practice is a fraction of a degree for a
     * mesh whose bounding box is already close to world-aligned.
     */
    void setOrientationMode(OrientationMode m);

    /** @copydoc setOrientationMode() */
    [[nodiscard]] auto orientationMode() const -> OrientationMode;

    /**
     * @brief Set explicit pinhole intrinsics/extrinsics for
     * ProjectionMode::Camera
     *
     * Stores an explicit camera that overrides the auto-derived one when
     * sampling in ProjectionMode::Camera. Does not change the projection mode;
     * use setProjectionMode() to enable camera sampling. Call
     * clearProjectionParams() to revert to the auto-derived camera.
     */
    void setProjectionParams(const ProjectionParams& params);

    /** @copydoc setProjectionParams() */
    [[nodiscard]] auto projectionParams() const -> ProjectionParams;

    /**
     * @brief Discard any explicit pinhole parameters
     *
     * After this call, ProjectionMode::Camera sampling reverts to the
     * auto-derived camera. Does not change the projection mode.
     */
    void clearProjectionParams();

    /** @brief Generate the new texture image and UV map */
    auto compute() -> cv::Mat;

    /** @brief Get the output UV map */
    auto getUVMap() -> UVMap;

    /**
     * @brief Get depth map
     *
     * Single-channel float image (CV_32FC1) in mesh units. For
     * ProjectionMode::Orthographic this is the distance from the sampling plane
     * along the mesh's shortest axis; for ProjectionMode::Camera it is the
     * perpendicular (optical-axis) depth, i.e. camera-space Z. Pixels with no
     * surface intersection are NaN.
     */
    auto getDepthMap() -> cv::Mat;

    /**
     * @brief Get the per-pixel 3D surface position map
     *
     * Three-channel float image (CV_32FC3) giving the (x, y, z) coordinate of
     * the surface point sampled by each pixel. ProjectionMode::Camera positions
     * are in the mesh's world frame; ProjectionMode::Orthographic positions are
     * in the realigned sampling frame. Pixels with no intersection are NaN.
     */
    auto getPositionMap() -> cv::Mat;

private:
    /** Resample the input image into the organized texture (orthographic) */
    void create_texture_();

    /** Resample the input image using a pinhole camera projection */
    void create_texture_camera_();

    /**
     * Resolve the input texture image a face samples from. Returns the image
     * for the face's UV chart, or nullptr if that chart has no usable image
     * (chart index out of range or an empty image); the offending chart index
     * is recorded in missingCharts_ for a single aggregated warning.
     */
    [[nodiscard]] auto resolve_chart_image_(std::size_t cellId) const
        -> const cv::Mat*;

    /**
     * Bilinearly sample @p img for a ray hit on face @p cellId with barycentric
     * intersection (@p interU, @p interV). Assumes @p img is non-empty.
     */
    [[nodiscard]] auto sample_surface_color_(
        const cv::Mat& img, std::size_t cellId, double interU, double interV)
        const -> cv::Vec3b;

    /** Emit one aggregated warning for charts with no usable image, if any */
    void report_missing_charts_() const;

    /** Input mesh */
    Mesh::Pointer inputMesh_;
    /** Input UV map */
    UVMap inputUV_;
    /** Input texture images, indexed by UV chart */
    std::vector<cv::Mat> inputTextures_;
    /** Chart indices encountered with no usable image (for warning) */
    mutable std::vector<std::size_t> missingCharts_;

    /** Sample origin */
    SamplingOrigin sampleOrigin_{SamplingOrigin::TopLeft};
    /** Sample mode */
    SamplingMode sampleMode_{SamplingMode::Rate};
    /** XY plane sample rate (in mesh units) */
    double sampleRate_{DEFAULT_SAMPLE_RATE};
    /** Length of the predefined sampling dimension */
    std::size_t sampleDim_{800};

    /** Whether we want the first or last mesh intersection point */
    bool useFirstIntersection_{false};

    /** Projection model */
    ProjectionMode projectionMode_{ProjectionMode::Orthographic};
    /** Sampling frame orientation resolution */
    OrientationMode orientationMode_{OrientationMode::OBB};
    /** Explicit pinhole parameters (used when projParamsSet_ is true) */
    ProjectionParams projParams_{};
    /** Whether explicit projection parameters were provided */
    bool projParamsSet_{false};

    /** Output UV map */
    UVMap outputUV_;
    /** Output texture image */
    cv::Mat outputTexture_;
    /** Output depth map */
    cv::Mat outputDepthMap_;
    /** Output per-pixel 3D position map */
    cv::Mat outputPositionMap_;
};

/**
 * @brief Validate pinhole camera parameters
 *
 * Returns a message describing the first problem found, or std::nullopt if
 * @p params describe a usable pinhole camera: positive, finite focal lengths
 * and image size; a finite principal point; and an orthonormal, right-handed
 * rotation block in the world-to-camera extrinsics.
 */
[[nodiscard]] auto ValidateProjectionParams(
    const ReorderUnorganizedTexture::ProjectionParams& params)
    -> std::optional<std::string>;

/**
 * @brief Apply the radial distortion model to a normalized image coordinate
 *
 * Maps an ideal (pinhole) normalized coordinate to its distorted location
 * using the radial model `r' = 1 + k1*r^2 + k2*r^4 + k3*r^6` (OpenMVG
 * @c pinhole_radial_k3). With all coefficients zero this is the identity.
 */
[[nodiscard]] auto DistortNormalized(
    const ReorderUnorganizedTexture::ProjectionParams& params,
    const cv::Vec2d& normalized) -> cv::Vec2d;

/**
 * @brief Invert the radial distortion model for a normalized image coordinate
 *
 * Recovers the ideal (pinhole) normalized coordinate from a distorted one by
 * fixed-point iteration (OpenCV's @c undistortPoints approach). With all
 * coefficients zero this is the identity.
 */
[[nodiscard]] auto UndistortNormalized(
    const ReorderUnorganizedTexture::ProjectionParams& params,
    const cv::Vec2d& distorted) -> cv::Vec2d;
}  // namespace rt
