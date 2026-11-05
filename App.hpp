#pragma once
// Native window, settings panel, camera and mouse-source interaction, status line and CSV export.
#include "GpuApi.hpp"
#include "Math3D.hpp"
#include "Model.hpp"
#include "Simulation.hpp"
#include <array>
#include <memory>

namespace optics {
constexpr int panelWidth = 320;   // settings column
constexpr int statusHeight = 64;  // status strip under the canvas
constexpr int sliderCount = 5, comboCount = 5;
constexpr int initialWidth = 1360, initialHeight = 1000, minimumWidth = 1080, minimumHeight = 760;
constexpr const wchar_t *appTitle = L"Wave Optics Lab", *appSubtitle = L"Diffraction and interference";

/// Control identifiers. Combos and sliders are consecutive so the index is id - first.
enum class Id : int {
    experiment = 10, grid, display, slicePlane, illumination, //
    slider = 20,                                              // slider + 0..4
    pause = 30, reset, step, resetCamera, exportCsv,          //
    absorbing = 40
};
enum class Slider : int { wavelength, opening, material, slice, exposure };

struct App {
    GpuApi gpu;
    std::unique_ptr<Simulation> sim;
    Camera camera;
    HWND window{}, status{}, pauseButton{}, absorbingBox{};
    std::array<HWND, comboCount> combos{};
    std::array<HWND, sliderCount> sliders{}, labels{};
    int width = initialWidth, height = initialHeight; // client size
    bool running = true, paused = false, single = false, orbiting = false, painting = false;
    POINT previous{}; // last mouse position while orbiting
    float fps = 0;

    App() = default;
    App(const App &) = delete;
    App &operator=(const App &) = delete;
    ~App();

    void create(HINSTANCE instance);
    void createControls();
    void resetCamera();
    /// Applies the slider positions to the settings; restarts the field when the geometry changed.
    void readSliders(bool allowReset = true);
    void resize();
    void togglePause();
    void advanceOne();
    void command(Id id, int code);
    bool key(WPARAM pressed);
    /// Dispatches pending window messages; returns false once the application should exit.
    bool pump();
    Viewport canvas() const;
    bool inCanvas(int x, int y) const;
    V3 ray(int x, int y) const;
    void placeSource(int x, int y);
    void mouse(int x, int y);
    void stopInteraction();
    void draw();
    void updateStatus();
    void exportDialog();
};
} // namespace optics
