#include "App.hpp"
#include "ModernUI.hpp"
#include <commctrl.h>
#include <commdlg.h>
#include <windowsx.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace optics {
namespace {
constexpr int margin = 20, contentWidth = 280;      // panel content spans x = 20..300
constexpr int buttonWidth = 130, buttonHeight = 32; // two buttons per row: 20..150 and 170..300
constexpr float orbitSpeed = .007f, zoomSpeed = .001f, minDistance = 1.25f, maxDistance = 4.5f;
constexpr std::array gridOptions{64, 96, 128};
constexpr std::array<int, sliderCount> initialPositions{35, 30, 40, 50, 35};
// The detector inset occupies this part of the canvas (normalized, origin bottom-left); see the shader.
constexpr float insetLeft = .73f, insetRight = .98f, insetBottom = .025f, insetTop = .31f;

int comboIndex(Id id) {
    return std::to_underlying(id) - std::to_underlying(Id::experiment);
}
HWND sliderOf(const App &app, Slider slider) {
    return app.sliders[static_cast<size_t>(std::to_underlying(slider))];
}
float position(HWND slider) {
    return static_cast<float>(SendMessageW(slider, TBM_GETPOS, 0, 0)) / 100;
}

LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM w, LPARAM l) {
    auto *app = reinterpret_cast<App *>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (message == WM_NCCREATE) {
        app = static_cast<App *>(reinterpret_cast<CREATESTRUCTW *>(l)->lpCreateParams);
        app->window = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
    }
    if (message == WM_GETMINMAXINFO) {
        reinterpret_cast<MINMAXINFO *>(l)->ptMinTrackSize = {minimumWidth, minimumHeight};
        return 0;
    }
    if (!app)
        return DefWindowProcW(hwnd, message, w, l);
    const int x = GET_X_LPARAM(l), y = GET_Y_LPARAM(l);
    switch (message) {
    case WM_CLOSE:
        app->running = false;
        return 0;
    case WM_SIZE:
        app->resize();
        return 0;
    case WM_ERASEBKGND:
        return 1; // the canvas and the panel are painted by OpenGL and the themed controls
    case WM_COMMAND:
        app->command(static_cast<Id>(LOWORD(w)), HIWORD(w));
        return 0;
    case WM_HSCROLL:
        app->readSliders();
        return 0;
    case WM_LBUTTONDOWN:
        if (app->inCanvas(x, y)) {
            app->painting = true;
            SetCapture(hwnd);
            app->placeSource(x, y);
        }
        return 0;
    case WM_RBUTTONDOWN:
        if (app->inCanvas(x, y)) {
            app->orbiting = true;
            app->previous = {x, y};
            SetCapture(hwnd);
        }
        return 0;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_CAPTURECHANGED:
        app->stopInteraction();
        return 0;
    case WM_MOUSEMOVE:
        app->mouse(x, y);
        return 0;
    case WM_MOUSEWHEEL:
        app->camera.distance = std::clamp(app->camera.distance * std::exp(-GET_WHEEL_DELTA_WPARAM(w) * zoomSpeed),
                                          minDistance, maxDistance);
        return 0;
    }
    return DefWindowProcW(hwnd, message, w, l);
}
} // namespace

App::~App() {
    sim.reset();    // GPU objects need the context
    gpu.shutdown(); // the context needs the window's DC
    if (window)
        DestroyWindow(window);
}

void App::create(HINSTANCE instance) {
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = windowProc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);
    wc.lpszClassName = L"WaveOptics3DGPU";
    if (!RegisterClassExW(&wc))
        throw std::runtime_error("Window registration failed.");
    if (!CreateWindowExW(0, wc.lpszClassName, appTitle, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT,
                         CW_USEDEFAULT, initialWidth, initialHeight, nullptr, nullptr, instance, this))
        throw std::runtime_error("Window creation failed.");
    gpu.initialize(window);
    sim = std::make_unique<Simulation>(gpu);
    createControls();
    modern::apply(window, {.sidebarWidth = panelWidth,
                           .designWidth = panelWidth,
                           .title = appTitle,
                           .subtitle = appSubtitle,
                           .primaryButtonId = std::to_underlying(Id::pause)});
    resize();
}

void App::createControls() {
    auto control = [&](const wchar_t *kind, const wchar_t *text, DWORD style, int x, int y, int w, int h,
                       int id = 0) {
        return CreateWindowExW(0, kind, text, WS_CHILD | WS_VISIBLE | style, x, y, w, h, window,
                               reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), GetModuleHandleW(nullptr),
                               nullptr);
    };
    auto combo = [&](const wchar_t *label, int y, Id id, std::initializer_list<const wchar_t *> items,
                     int initial) {
        control(L"STATIC", label, 0, margin, y, contentWidth, 21);
        HWND c = control(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, margin, y + 23, contentWidth, 180,
                         std::to_underlying(id));
        for (auto item : items)
            SendMessageW(c, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(item));
        SendMessageW(c, CB_SETCURSEL, static_cast<WPARAM>(initial), 0);
        return c;
    };
    auto button = [&](const wchar_t *text, int x, int y, int w, Id id) {
        return control(L"BUTTON", text, BS_PUSHBUTTON | WS_TABSTOP, x, y, w, buttonHeight, std::to_underlying(id));
    };

    control(L"STATIC", appTitle, 0, margin, 14, contentWidth, 27);
    control(L"STATIC", appSubtitle, 0, margin, 43, contentWidth, 24);
    combos[comboIndex(Id::experiment)] = combo(L"Experiment (resets)", 76, Id::experiment,
                                               {L"Double rectangular slit", L"Circular aperture", L"Refractive ball lens"}, 0);
    combos[comboIndex(Id::grid)] =
        combo(L"3D grid (resets)", 139, Id::grid,
              {L"64 x 64 x 64 / fast", L"96 x 96 x 96 / balanced", L"128 x 128 x 128 / detailed"}, 1);
    combos[comboIndex(Id::display)] =
        combo(L"Display", 202, Id::display,
              {L"Volume / signed amplitude", L"Volume / mean intensity", L"Slice / signed amplitude",
               L"Slice / mean intensity", L"Detector / Y-Z plane"},
              0);
    combos[comboIndex(Id::slicePlane)] = combo(L"Slice and mouse-source plane", 265, Id::slicePlane,
                                               {L"XY / move along Z", L"XZ / move along Y", L"YZ / move along X"}, 0);
    combos[comboIndex(Id::illumination)] =
        combo(L"Incident illumination", 328, Id::illumination,
              {L"Continuous coherent beam", L"Gaussian pulse (restart)", L"Source off / let waves decay"}, 0);
    for (int i = 0; i < sliderCount; ++i) {
        const int y = 393 + i * 58;
        labels[i] = control(L"STATIC", L"", 0, margin, y, contentWidth, 23);
        sliders[i] = control(TRACKBAR_CLASSW, L"", TBS_HORZ | WS_TABSTOP, margin - 6, y + 24, contentWidth + 12, 30,
                             std::to_underlying(Id::slider) + i);
        SendMessageW(sliders[i], TBM_SETRANGE, TRUE, MAKELPARAM(0, 100));
        SendMessageW(sliders[i], TBM_SETPOS, TRUE, initialPositions[i]);
    }
    absorbingBox = control(L"BUTTON", L"Absorbing boundary sponge", BS_AUTOCHECKBOX | WS_TABSTOP, margin, 686,
                           contentWidth, 26, std::to_underlying(Id::absorbing));
    SendMessageW(absorbingBox, BM_SETCHECK, BST_CHECKED, 0);
    const int secondColumn = margin + contentWidth - buttonWidth;
    pauseButton = button(L"Pause", margin, 722, buttonWidth, Id::pause);
    button(L"Reset", secondColumn, 722, buttonWidth, Id::reset);
    button(L"Single step", margin, 764, buttonWidth, Id::step);
    button(L"Reset camera", secondColumn, 764, buttonWidth, Id::resetCamera);
    control(L"BUTTON", L"Export detector CSV...", BS_PUSHBUTTON | WS_TABSTOP, margin, 806, contentWidth, 30,
            std::to_underlying(Id::exportCsv));
    control(L"STATIC",
            L"LMB: source on selected slice\nRMB: orbit   Wheel: zoom\nSpace: pause   R: reset   N: step\n"
            L"Detector inset: Y up, Z right",
            0, margin, 849, contentWidth, 78);
    status = control(L"STATIC", L"Initializing...", 0, panelWidth + 12, height - 58, width - panelWidth - 24, 54);
    readSliders();
    resetCamera();
}

void App::resetCamera() {
    camera = Camera{.yaw = .65f, .pitch = .35f, .distance = 2.65f, .target = {.5f, .5f, .5f}};
}

void App::readSliders(bool allowReset) {
    if (!sim)
        return;
    auto &s = sim->settings;
    const Settings before = s;
    const float material = position(sliderOf(*this, Slider::material));
    s.refractiveIndex = 1 + material;
    // The wavelength range keeps at least minCellsPerWavelength cells per local wavelength.
    const float shortest = minCellsPerWavelength * (s.scene == Scene::ballLens ? s.refractiveIndex : 1.f) /
                           static_cast<float>(s.n - 1);
    const float longest = std::max(.24f, shortest + .08f);
    s.wavelength = shortest + (longest - shortest) * position(sliderOf(*this, Slider::wavelength));
    const float opening = position(sliderOf(*this, Slider::opening));
    switch (s.scene) {
    case Scene::doubleSlit:
        s.opening = .04f + .12f * opening;
        break;
    case Scene::circularAperture:
        s.opening = .075f + .15f * opening;
        break;
    case Scene::ballLens:
    default:
        s.opening = .12f + .15f * opening;
        break;
    }
    s.separation = std::max(s.opening + 2.f / static_cast<float>(s.n - 1), .16f + .3f * material);
    s.slice = .12f + .76f * position(sliderOf(*this, Slider::slice));
    s.exposure = .5f + 7 * position(sliderOf(*this, Slider::exposure));

    const wchar_t *openingLabel = s.scene == Scene::doubleSlit        ? L"Slit width"
                                  : s.scene == Scene::circularAperture ? L"Aperture radius"
                                                                        : L"Lens radius";
    const std::wstring materialText = s.scene == Scene::ballLens     ? std::format(L"Refractive index: {:.2f}", s.refractiveIndex)
                                      : s.scene == Scene::doubleSlit ? std::format(L"Slit separation: {:.3f}", s.separation)
                                                                     : std::wstring(L"Separation: double-slit mode only");
    const std::array<std::wstring, sliderCount> text{
        std::format(L"Vacuum wavelength: {:.3f}", s.wavelength), std::format(L"{}: {:.3f}", openingLabel, s.opening),
        materialText, std::format(L"Slice position: {:.2f}", s.slice), std::format(L"Display exposure: {:.2f}", s.exposure)};
    for (int i = 0; i < sliderCount; ++i)
        SetWindowTextW(labels[i], text[static_cast<size_t>(i)].c_str());
    if (allowReset && s.resetsOnChange(before)) {
        stopInteraction();
        sim->reset();
    }
}

void App::resize() {
    RECT r{};
    GetClientRect(window, &r);
    width = r.right;
    height = r.bottom;
    if (status)
        MoveWindow(status, panelWidth + 12, height - 58, std::max(1, width - panelWidth - 24), 54, TRUE);
}

void App::togglePause() {
    paused = !paused;
    SetWindowTextW(pauseButton, paused ? L"Resume" : L"Pause");
}

void App::advanceOne() {
    sim->tick(1);
    single = false;
}

void App::command(Id id, int code) {
    if (!sim)
        return;
    auto &s = sim->settings;
    switch (id) {
    case Id::experiment:
    case Id::grid:
    case Id::display:
    case Id::slicePlane:
    case Id::illumination: {
        if (code != CBN_SELCHANGE)
            return;
        const int v = static_cast<int>(SendMessageW(combos[static_cast<size_t>(comboIndex(id))], CB_GETCURSEL, 0, 0));
        if (v < 0)
            return;
        switch (id) {
        case Id::experiment:
            s.scene = static_cast<Scene>(v);
            break;
        case Id::grid:
            s.n = gridOptions[static_cast<size_t>(v)];
            break;
        case Id::display:
            s.display = static_cast<Display>(v);
            return;
        case Id::slicePlane:
            stopInteraction();
            s.sliceAxis = 2 - v; // XY moves along Z, XZ along Y, YZ along X
            return;
        case Id::illumination:
            s.source = static_cast<Source>(v);
            if (s.source != Source::off) { // switching the source off keeps the existing waves
                stopInteraction();
                sim->reset();
            }
            return;
        default:
            return;
        }
        stopInteraction();
        readSliders(false); // the scene and grid change the slider ranges
        sim->reset();
        return;
    }
    case Id::pause:
        togglePause();
        return;
    case Id::reset:
        stopInteraction();
        sim->reset();
        return;
    case Id::step:
        paused = true;
        single = true;
        SetWindowTextW(pauseButton, L"Resume");
        return;
    case Id::resetCamera:
        stopInteraction();
        resetCamera();
        return;
    case Id::absorbing:
        s.absorbing = SendMessageW(absorbingBox, BM_GETCHECK, 0, 0) == BST_CHECKED;
        stopInteraction();
        sim->reset();
        return;
    case Id::exportCsv:
        exportDialog();
        return;
    default:
        return;
    }
}

bool App::key(WPARAM pressed) {
    switch (pressed) {
    case VK_ESCAPE:
        running = false;
        return true;
    case VK_SPACE:
        togglePause();
        return true;
    case 'R':
        command(Id::reset, 0);
        return true;
    case 'N':
        command(Id::step, 0);
        return true;
    default:
        return false;
    }
}

bool App::pump() {
    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT)
            running = false;
        if (msg.message == WM_KEYDOWN && msg.hwnd == window && key(msg.wParam))
            continue; // simulation shortcuts apply while the canvas has focus
        if (IsDialogMessageW(window, &msg))
            continue; // Tab navigation between the panel controls
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return running;
}

Viewport App::canvas() const {
    return {panelWidth, statusHeight, std::max(1, width - panelWidth), std::max(1, height - statusHeight)};
}

bool App::inCanvas(int x, int y) const {
    return x > panelWidth && x < width && y >= 0 && y < height - statusHeight;
}

V3 App::ray(int x, int y) const {
    const auto c = canvas();
    const float w = static_cast<float>(c.width), h = static_cast<float>(c.height);
    return camera.ray(2 * static_cast<float>(x - panelWidth) / w - 1, 1 - 2 * static_cast<float>(y) / h, w / h);
}

void App::placeSource(int x, int y) {
    if (!sim)
        return;
    auto &s = sim->settings;
    // Intersect the cursor ray with the slice plane; the source follows it while held.
    const V3 d = ray(x, y), e = camera.eye();
    const int axis = s.sliceAxis;
    const float t = std::abs(d[axis]) > 1e-6f ? (s.slice - e[axis]) / d[axis] : -1;
    const V3 p = e + d * t;
    const auto c = canvas();
    const float u = static_cast<float>(x - panelWidth) / static_cast<float>(c.width);
    const float v = 1 - static_cast<float>(y) / static_cast<float>(c.height);
    const bool overInset = u > insetLeft && u < insetRight && v > insetBottom && v < insetTop;
    const bool insideDomain = p.x > .02f && p.x < .98f && p.y > .02f && p.y < .98f && p.z > .02f && p.z < .98f;
    s.brushActive = inCanvas(x, y) && !overInset && s.display != Display::detector && t > 0 && insideDomain;
    if (s.brushActive)
        s.brush = p;
}

void App::mouse(int x, int y) {
    if (orbiting) {
        camera.yaw += static_cast<float>(x - previous.x) * orbitSpeed;
        camera.pitch = std::clamp(camera.pitch + static_cast<float>(y - previous.y) * orbitSpeed, -maxPitch, maxPitch);
    }
    if (painting)
        placeSource(x, y);
    previous = {x, y};
}

void App::stopInteraction() {
    painting = false;
    orbiting = false;
    if (sim)
        sim->settings.brushActive = false;
    if (GetCapture() == window)
        ReleaseCapture();
}

void App::draw() {
    gpu.BindFramebuffer(gl::FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    glClearColor(.012f, .018f, .032f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    // The header above the settings panel is painted by OpenGL too.
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, panelWidth, height);
    glClearColor(246.f / 255, 248.f / 255, 252.f / 255, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    sim->draw(camera, canvas());
}

void App::updateStatus() {
    sim->measure();
    const bool healthy = sim->healthy();
    if (!healthy) {
        paused = true;
        SetWindowTextW(pauseButton, L"Resume");
    }
    const auto &s = sim->settings;
    const auto text = std::format(L"{} cubed / {} cells | GPU {:.2f} ms | {:.0f} fps | time {:.3f} | CFL {:.2f}\n"
                                  L"Min wavelength {:.1f} cells | max |u| {:.3f} | detector peak {:.5f}{}",
                                  s.n, sim->cells(), sim->gpuMs, fps, sim->time, courant, s.cellsPerWavelength(),
                                  sim->metrics.maxAmplitude(), sim->metrics.detectorPeak(),
                                  healthy ? L"" : L" | INVALID: reset required");
    SetWindowTextW(status, text.c_str());
}

void App::exportDialog() {
    wchar_t filename[MAX_PATH] = L"WaveOptics-detector.csv";
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = window;
    dialog.lpstrFilter = L"CSV files\0*.csv\0All files\0*.*\0";
    dialog.lpstrFile = filename;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrDefExt = L"csv";
    dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameW(&dialog))
        return;
    try {
        sim->exportDetector(filename);
    } catch (const std::exception &e) {
        MessageBoxA(window, e.what(), "Detector export", MB_ICONERROR);
    }
}
} // namespace optics
