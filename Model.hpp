#pragma once
// Model constants and the GPU-visible records shared by the shaders, the simulation and the UI.
// The std140 structs mirror the uniform blocks in OpticsShaders.hpp; the static_asserts pin
// their layout.
#include "Math3D.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace optics {
using F4 = std::array<float, 4>;

constexpr int minGrid = 16, maxGrid = 128;
constexpr float courant = .45f;             // dt = courant * h; below the cubic-grid limit 1/sqrt(3)
constexpr float detectorPlane = .82f;       // X of the Y-Z detector; keep in sync with screenX() in GLSL
constexpr float minCellsPerWavelength = 8;  // the wavelength slider never goes below this sampling
constexpr int tickRate = 60;                // nominal application updates per second
constexpr int workgroupSize = 128;          // layout(local_size_x) of every kernel

constexpr int workgroups(int invocations) {
    return (invocations + workgroupSize - 1) / workgroupSize;
}

enum class Scene : int { doubleSlit, circularAperture, ballLens };
enum class Source : int { continuous, pulse, off };
enum class Display : int { volumeAmplitude, volumeIntensity, sliceAmplitude, sliceIntensity, detector };

struct Settings {
    int n = 96;                      // grid edge: 64, 96 or 128 in the UI
    Scene scene = Scene::doubleSlit;
    Source source = Source::continuous;
    Display display = Display::volumeAmplitude;
    int sliceAxis = 2;               // 0 = X, 1 = Y, 2 = Z: the axis the slice plane moves along
    float wavelength = .14f;         // vacuum wavelength, domain units
    float opening = .076f;           // slit width, aperture radius or lens radius
    float separation = .28f;         // slit separation (double slit only)
    float refractiveIndex = 1.4f;    // ball lens only
    float slice = .5f;               // slice position along sliceAxis
    float exposure = 2.95f;          // display only
    bool absorbing = true;           // damping sponge at the outer boundary
    bool brushActive = false;        // the mouse source is being held
    V3 brush{};

    int cells() const {
        return n * n * n;
    }
    float stepSize() const {
        return courant / static_cast<float>(n - 1);
    }
    /// Steps per nominal tick: the field crosses the domain in about the same real time at any grid.
    int stepsPerTick() const {
        return (n + 11) / 12;
    }
    bool volumeDisplay() const {
        return display < Display::sliceAmplitude;
    }
    float cellsPerWavelength() const {
        return wavelength * static_cast<float>(n - 1) / (scene == Scene::ballLens ? refractiveIndex : 1.f);
    }
    /// Changes to the optical geometry or the material restart the field.
    bool resetsOnChange(const Settings &other) const {
        return wavelength != other.wavelength || opening != other.opening ||
               (scene == Scene::doubleSlit && separation != other.separation) ||
               (scene == Scene::ballLens && refractiveIndex != other.refractiveIndex);
    }
};

/// 48-byte diagnostic summary produced by the GPU reductions (shaders::reduce / finalReduce).
struct alignas(16) Metrics {
    F4 totals{};  // x: discrete energy, y: Σ intensity h³, z: invalid records, w: Σ u² inside the opaque screen
    F4 bounds{};  // x: max |u|, y: max intensity, z: detector peak, w: max |u| away from the central Z band
    F4 spatial{}; // x: Σ intensity h³ downstream of x = 0.6, y: Σ u² h³

    float energy() const {
        return totals[0];
    }
    float invalid() const {
        return totals[2];
    }
    float screenLeak() const {
        return totals[3];
    }
    float maxAmplitude() const {
        return bounds[0];
    }
    float detectorPeak() const {
        return bounds[2];
    }
};

/// std140 mirror of shaders::paramsBlock (uniform binding 0).
struct alignas(16) Params {
    std::int32_t n, scene, sourceMode, absorbing;
    std::int32_t brushActive, display, sliceAxis;
    float dt;
    float time, wavelength, opening, separation;
    float refractiveIndex, exposure, slicePosition, padding0;
    V3 brush;
    float padding1;
};

/// std140 mirror of shaders::viewBlock (uniform binding 1).
struct alignas(16) ViewParams {
    F4 eye, forward, right, up;
    float aspect, tanHalfFov;
    float padding[2];
};

static_assert(sizeof(Metrics) == 48);
static_assert(sizeof(Params) == 80 && offsetof(Params, time) == 32 && offsetof(Params, brush) == 64);
static_assert(sizeof(ViewParams) == 80 && offsetof(ViewParams, aspect) == 64);

/// Window-space rectangle, in pixels from the bottom-left corner as OpenGL counts them.
struct Viewport {
    int x, y, width, height;
};
} // namespace optics
