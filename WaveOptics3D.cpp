#include "ModernUI.hpp"
#include "GpuApi.hpp"
#include "Math3D.hpp"
#include "OpticsShaders.hpp"
#include <commctrl.h>
#include <commdlg.h>
#include <windowsx.h>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>

namespace optics {
constexpr int panelWidth = 320;
struct Settings {
    int n = 96, scene = 0, source = 0, display = 0, sliceAxis = 2, fixture = 0;
    float wavelength = .14f, opening = .076f, separation = .28f, refractiveIndex = 1.4f, slice = .5f,
          exposure = 2.95f;
    bool absorbing = true, brushActive = false;
    V3 brush{};
};
struct alignas(16) Metric {
    std::array<float, 4> totals{}, bounds{}, spatial{};
};
static_assert(sizeof(Metric) == 48);
class Simulation {
    GL &gl;
    GLuint initialize{}, step{}, reduce{}, finalReduce{}, render{};
    GLuint current{}, previous{}, next{}, average{}, medium{}, reduction{}, screen{}, query{}, fbo{},
        texture{};
    bool queryPending = false;
    int imageWidth{}, imageHeight{};
    void uniforms(GLuint program) {
        gl.UseProgram(program);
        gl.integer(program, "n", settings.n);
        gl.integer(program, "scene", settings.scene);
        gl.integer(program, "sourceMode", settings.source);
        gl.integer(program, "absorbing", settings.absorbing);
        gl.integer(program, "fixture", settings.fixture);
        gl.integer(program, "brushActive", settings.brushActive);
        gl.integer(program, "display", settings.display);
        gl.integer(program, "sliceAxis", settings.sliceAxis);
        gl.integer(program, "showInset", 1);
        gl.scalar(program, "dt", stepSize());
        gl.scalar(program, "time", static_cast<float>(time));
        gl.scalar(program, "wavelength", settings.wavelength);
        gl.scalar(program, "opening", settings.opening);
        gl.scalar(program, "separation", settings.separation);
        gl.scalar(program, "refractiveIndex", settings.refractiveIndex);
        gl.scalar(program, "slicePosition", settings.slice);
        gl.scalar(program, "exposure", settings.exposure);
        gl.vector(program, "brush", settings.brush.x, settings.brush.y, settings.brush.z);
    }
    void allocate(GLuint buffer, size_t bytes) {
        gl.BindBuffer(SSBO, buffer);
        gl.BufferData(SSBO, static_cast<std::ptrdiff_t>(bytes), nullptr, 0x88E8);
    }
    void bindFields() {
        gl.BindBufferBase(SSBO, 0, current);
        gl.BindBufferBase(SSBO, 1, previous);
        gl.BindBufferBase(SSBO, 2, next);
    }
    void diagnostics() {
        uniforms(reduce);
        gl.dispatch(count());
        uniforms(finalReduce);
        gl.dispatch(128);
    }
    std::vector<float> read(GLuint buffer, int size) {
        std::vector<float> result(size);
        gl.MemoryBarrier(READBACK_BARRIER);
        gl.BindBuffer(SSBO, buffer);
        gl.GetBufferSubData(SSBO, 0, static_cast<std::ptrdiff_t>(result.size() * sizeof(float)),
                            result.data());
        return result;
    }

  public:
    Settings settings;
    Metric metrics;
    double time{};
    std::uint64_t steps{};
    float gpuMs{};
    explicit Simulation(GL &api) : gl(api) {
        auto compute = [&](const std::string &text, const char *label) {
            return gl.program({{COMPUTE, shaders::common + text}}, label);
        };
        initialize = compute(shaders::initialize, "Initialize optical domain");
        step = compute(shaders::step, "3D scalar wave FDTD");
        reduce = compute(shaders::reduce, "Wave diagnostics");
        finalReduce = compute(shaders::finalReduce, "Final wave reduction");
        render = gl.program({{VERTEX, shaders::vertex}, {FRAGMENT, shaders::common + shaders::fragment}},
                            "Volume, slice and detector renderer");
        current = gl.buffer(0, 16);
        previous = gl.buffer(1, 16);
        next = gl.buffer(2, 16);
        average = gl.buffer(3, 16);
        medium = gl.buffer(4, 16);
        reduction = gl.buffer(5, 48);
        screen = gl.buffer(6, 16);
        gl.GenQueries(1, &query);
        gl.GenFramebuffers(1, &fbo);
        glGenTextures(1, &texture);
        reset();
    }
    ~Simulation() {
        gl.DeleteQueries(1, &query);
        gl.DeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &texture);
    }
    int count() const {
        return settings.n * settings.n * settings.n;
    }
    float stepSize() const {
        return .45f / (settings.n - 1);
    }
    float cellsPerWavelength() const {
        return settings.wavelength * (settings.n - 1) /
               (settings.scene == 2 ? settings.refractiveIndex : 1.f);
    }
    void reset() {
        if (settings.n < 16 || settings.n > 128 || settings.wavelength <= 0 || settings.refractiveIndex < 1)
            throw std::runtime_error("Unsupported optical grid or material parameters.");
        time = 0;
        steps = 0;
        settings.brushActive = false;
        for (auto b : {current, previous, next, average})
            allocate(b, static_cast<size_t>(count()) * 4);
        allocate(medium, static_cast<size_t>(count()) * 8);
        allocate(reduction, static_cast<size_t>((count() + 127) / 128 + 1) * sizeof(Metric));
        allocate(screen, static_cast<size_t>(settings.n) * settings.n * 4);
        bindFields();
        uniforms(initialize);
        gl.dispatch(count());
        measure();
    }
    void tick(int iterations = 0) {
        if (iterations <= 0)
            iterations = (settings.n + 11) / 12;
        if (queryPending) {
            GLuint ready{};
            gl.GetQueryObjectuiv(query, 0x8867, &ready);
            if (ready) {
                unsigned long long ns{};
                gl.GetQueryObjectui64v(query, 0x8866, &ns);
                gpuMs = static_cast<float>(ns) * 1e-6f;
                queryPending = false;
            }
        }
        bool timing = !queryPending;
        if (timing)
            gl.BeginQuery(0x88BF, query);
        uniforms(step);
        for (int i = 0; i < iterations; ++i) {
            ++steps;
            time = steps * static_cast<double>(stepSize());
            gl.scalar(step, "time", static_cast<float>(time));
            gl.dispatch(count());
            auto spare = previous;
            previous = current;
            current = next;
            next = spare;
            bindFields();
        }
        diagnostics();
        if (timing) {
            gl.EndQuery(0x88BF);
            queryPending = true;
        }
    }
    void measure() {
        diagnostics();
        gl.MemoryBarrier(READBACK_BARRIER);
        gl.BindBuffer(SSBO, reduction);
        gl.GetBufferSubData(SSBO, 0, sizeof(metrics), &metrics);
    }
    bool healthy() const {
        return metrics.totals[2] == 0 && std::isfinite(metrics.totals[0]) && metrics.totals[0] > -1e-5f &&
               metrics.bounds[0] < 1000 && metrics.totals[3] == 0;
    }
    std::vector<float> snapshot() {
        return read(current, count());
    }
    std::vector<float> previousSnapshot() {
        return read(previous, count());
    }
    std::vector<float> intensitySnapshot() {
        return read(average, count());
    }
    std::vector<float> mediumSnapshot() {
        return read(medium, count() * 2);
    }
    std::vector<float> detectorSnapshot() {
        return read(screen, settings.n * settings.n);
    }
    void exportDetector(const std::filesystem::path &path) {
        auto data = detectorSnapshot();
        std::ofstream out(path);
        out << std::setprecision(9);
        out << "# Wave Optics Lab 3D; EMA of squared scalar amplitude; not calibrated radiance\n";
        out << "# n=" << settings.n << ", time=" << time << ", wavelength=" << settings.wavelength
            << ", x=" << std::round(.82 * (settings.n - 1)) / (settings.n - 1) << "\n";
        out << "y,z,mean_squared_amplitude\n";
        for (int z = 0; z < settings.n; ++z)
            for (int y = 0; y < settings.n; ++y)
                out << y / static_cast<double>(settings.n - 1) << ','
                    << z / static_cast<double>(settings.n - 1) << ',' << data[y + settings.n * z] << '\n';
        if (!out)
            throw std::runtime_error("Unable to write detector CSV.");
    }
    void draw(const Camera &camera, int width, int height) {
        int w = settings.display < 2 ? (width + 1) / 2 : width,
            h = settings.display < 2 ? (height + 1) / 2 : height;
        gl.BindFramebuffer(0x8D40, fbo);
        if (w != imageWidth || h != imageHeight) {
            imageWidth = w;
            imageHeight = h;
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F);
            glTexImage2D(GL_TEXTURE_2D, 0, 0x8058, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            gl.FramebufferTexture2D(0x8D40, 0x8CE0, GL_TEXTURE_2D, texture, 0);
            if (gl.CheckFramebufferStatus(0x8D40) != 0x8CD5)
                throw std::runtime_error("Volume render framebuffer is incomplete.");
        }
        glViewport(0, 0, w, h);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_SCISSOR_TEST);
        uniforms(render);
        gl.scalar(render, "aspect", static_cast<float>(width) / height);
        auto eye = camera.eye(), forward = camera.forward(), right = camera.right(), up = camera.up();
        gl.vector(render, "eye", eye.x, eye.y, eye.z);
        gl.vector(render, "cameraForward", forward.x, forward.y, forward.z);
        gl.vector(render, "cameraRight", right.x, right.y, right.z);
        gl.vector(render, "cameraUp", up.x, up.y, up.z);
        gl.BindVertexArray(gl.vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        gl.BindFramebuffer(0x8CA8, fbo);
        gl.BindFramebuffer(0x8CA9, 0);
        gl.BlitFramebuffer(0, 0, w, h, panelWidth, 64, panelWidth + width, 64 + height, GL_COLOR_BUFFER_BIT,
                           GL_LINEAR);
        gl.BindFramebuffer(0x8D40, 0);
    }
};

void capture(const std::filesystem::path &path, int x, int y, int width, int height) {
    int stride = (width * 3 + 3) & ~3;
    std::vector<unsigned char> pixels(static_cast<size_t>(stride) * height);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(x, y, width, height, GL_BGR_EXT, GL_UNSIGNED_BYTE, pixels.data());
    BITMAPFILEHEADER header{};
    BITMAPINFOHEADER info{};
    header.bfType = 0x4d42;
    header.bfOffBits = sizeof(header) + sizeof(info);
    header.bfSize = header.bfOffBits + static_cast<DWORD>(pixels.size());
    info.biSize = sizeof(info);
    info.biWidth = width;
    info.biHeight = height;
    info.biPlanes = 1;
    info.biBitCount = 24;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    out.write(reinterpret_cast<const char *>(&info), sizeof(info));
    out.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
    if (!out)
        throw std::runtime_error("Unable to save framebuffer image.");
}

struct App {
    GL gl;
    std::unique_ptr<Simulation> sim;
    Camera camera;
    HWND window{}, status{}, pause{}, combos[5]{}, sliders[5]{}, labels[5]{}, absorber{};
    HFONT font{};
    int width = 1320, height = 920;
    bool running = true, paused = false, single = false, orbit = false, dragging = false;
    POINT previous{};
    float fps{};
    ~App() {
        sim.reset();
        if (font)
            DeleteObject(font);
    }
    HWND control(const wchar_t *kind, const std::wstring &text, DWORD style, int x, int y, int w, int h,
                 int id = 0) {
        auto item = CreateWindowExW(0, kind, text.c_str(), WS_CHILD | WS_VISIBLE | style, x, y, w, h, window,
                                    reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                    GetModuleHandleW(nullptr), nullptr);
        SendMessageW(item, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return item;
    }
    HWND combo(const wchar_t *label, int y, int id, std::initializer_list<const wchar_t *> items) {
        control(L"STATIC", label, 0, 18, y, 250, 21);
        auto c = control(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, 18, y + 23, 250, 180, id);
        for (auto item : items)
            SendMessageW(c, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(item));
        SendMessageW(c, CB_SETCURSEL, 0, 0);
        return c;
    }
    void controls() {
        font = CreateFontW(-16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                           CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        control(L"STATIC", L"WAVE OPTICS LAB / 3D GPU", 0, 18, 14, 250, 27);
        control(L"STATIC", L"Diffraction, interference, refraction", 0, 18, 43, 250, 24);
        combos[0] = combo(L"Experiment (resets)", 76, 10,
                          {L"Double rectangular slit", L"Circular aperture", L"Refractive ball lens"});
        combos[1] =
            combo(L"3D grid (resets)", 139, 11,
                  {L"64 x 64 x 64 / fast", L"96 x 96 x 96 / balanced", L"128 x 128 x 128 / detailed"});
        SendMessageW(combos[1], CB_SETCURSEL, 1, 0);
        combos[2] = combo(L"Display", 202, 12,
                          {L"Volume / signed amplitude", L"Volume / mean intensity",
                           L"Slice / signed amplitude", L"Slice / mean intensity", L"Detector / Y-Z plane"});
        combos[3] = combo(L"Slice and mouse-source plane", 265, 13,
                          {L"XY / move along Z", L"XZ / move along Y", L"YZ / move along X"});
        combos[4] = combo(
            L"Incident illumination", 328, 14,
            {L"Continuous coherent beam", L"Gaussian pulse (restart)", L"Source off / let waves decay"});
        int positions[] = {35, 30, 40, 50, 35};
        for (int i = 0; i < 5; ++i) {
            labels[i] = control(L"STATIC", L"", 0, 18, 393 + i * 58, 250, 23);
            sliders[i] =
                control(TRACKBAR_CLASSW, L"", TBS_HORZ | WS_TABSTOP, 12, 417 + i * 58, 260, 30, 20 + i);
            SendMessageW(sliders[i], TBM_SETRANGE, TRUE, MAKELPARAM(0, 100));
            SendMessageW(sliders[i], TBM_SETPOS, TRUE, positions[i]);
        }
        absorber = control(L"BUTTON", L"Absorbing boundary sponge", BS_AUTOCHECKBOX | WS_TABSTOP, 18, 686,
                           250, 26, 40);
        SendMessageW(absorber, BM_SETCHECK, BST_CHECKED, 0);
        pause = control(L"BUTTON", L"Pause", BS_PUSHBUTTON | WS_TABSTOP, 18, 722, 116, 32, 30);
        control(L"BUTTON", L"Reset", BS_PUSHBUTTON | WS_TABSTOP, 148, 722, 120, 32, 31);
        control(L"BUTTON", L"Single step", BS_PUSHBUTTON | WS_TABSTOP, 18, 764, 116, 32, 32);
        control(L"BUTTON", L"Reset camera", BS_PUSHBUTTON | WS_TABSTOP, 148, 764, 120, 32, 33);
        control(L"BUTTON", L"Export detector CSV...", BS_PUSHBUTTON | WS_TABSTOP, 18, 806, 250, 30, 34);
        control(L"STATIC",
                L"LMB: source on selected slice\nRMB: orbit   Wheel: zoom\nSpace: pause   R: reset   N: "
                L"step\nDetector inset: Y up, Z right",
                0, 18, 849, 252, 78);
        status = control(L"STATIC", L"Initializing...", 0, panelWidth + 12, height - 58,
                         width - panelWidth - 24, 54);
        updateSliders();
    }
    void updateSliders() {
        if (!sim)
            return;
        int v[5]{};
        for (int i = 0; i < 5; ++i)
            v[i] = static_cast<int>(SendMessageW(sliders[i], TBM_GETPOS, 0, 0));
        auto &s = sim->settings;
        auto old = s;
        s.refractiveIndex = 1 + v[2] * .01f;
        float minimum = 8 * (s.scene == 2 ? s.refractiveIndex : 1.f) / (s.n - 1),
              maximum = std::max(.24f, minimum + .08f);
        s.wavelength = minimum + (maximum - minimum) * v[0] * .01f;
        s.opening = s.scene == 0 ? .04f + v[1] * .0012f
                                 : (s.scene == 1 ? .075f + v[1] * .0015f : .12f + v[1] * .0015f);
        s.separation = std::max(s.opening + 2.f / (s.n - 1), .16f + v[2] * .003f);
        s.slice = .12f + v[3] * .0076f;
        s.exposure = .5f + v[4] * .07f;
        std::wstring text[] = {
            std::format(L"Vacuum wavelength: {:.3f}", s.wavelength),
            std::format(L"{}: {:.3f}",
                        s.scene == 0 ? L"Slit width" : (s.scene == 1 ? L"Aperture radius" : L"Lens radius"),
                        s.opening),
            s.scene == 2 ? std::format(L"Refractive index: {:.2f}", s.refractiveIndex)
                         : (s.scene == 0 ? std::format(L"Slit separation: {:.3f}", s.separation)
                                         : L"Separation: double-slit mode only"),
            std::format(L"Slice position: {:.2f}", s.slice),
            std::format(L"Display exposure: {:.2f}", s.exposure)};
        for (int i = 0; i < 5; ++i)
            SetWindowTextW(labels[i], text[i].c_str());
        if (s.wavelength != old.wavelength || s.opening != old.opening ||
            (s.scene == 0 && s.separation != old.separation) ||
            (s.scene == 2 && s.refractiveIndex != old.refractiveIndex)) {
            stop();
            sim->reset();
        }
    }
    void resize() {
        RECT r{};
        GetClientRect(window, &r);
        width = r.right;
        height = r.bottom;
        if (status)
            MoveWindow(status, panelWidth + 12, height - 58, std::max(1, width - panelWidth - 24), 54, TRUE);
    }
    void toggle() {
        paused = !paused;
        SetWindowTextW(pause, paused ? L"Resume" : L"Pause");
    }
    void advanceOne() {
        sim->tick(1);
        single = false;
    }
    void command(int id, int code) {
        if (!sim)
            return;
        if (id >= 10 && id <= 14 && code == CBN_SELCHANGE) {
            int v = static_cast<int>(SendMessageW(combos[id - 10], CB_GETCURSEL, 0, 0));
            if (id == 10)
                sim->settings.scene = v;
            if (id == 11)
                sim->settings.n = std::array{64, 96, 128}[v];
            if (id == 12)
                sim->settings.display = v;
            if (id == 13) {
                stop();
                sim->settings.sliceAxis = 2 - v;
            }
            if (id == 14) {
                sim->settings.source = v;
                if (v != 2) {
                    stop();
                    sim->reset();
                }
            }
            if (id <= 11) {
                stop();
                updateSliders();
                sim->reset();
            }
        }
        if (id == 30)
            toggle();
        if (id == 31) {
            stop();
            sim->reset();
        }
        if (id == 32) {
            paused = true;
            single = true;
            SetWindowTextW(pause, L"Resume");
        }
        if (id == 33) {
            stop();
            camera = Camera{};
        }
        if (id == 40) {
            sim->settings.absorbing = SendMessageW(absorber, BM_GETCHECK, 0, 0) == BST_CHECKED;
            stop();
            sim->reset();
        }
        if (id == 34) {
            wchar_t filename[MAX_PATH] = L"WaveOptics-detector.csv";
            OPENFILENAMEW dialog{};
            dialog.lStructSize = sizeof(dialog);
            dialog.hwndOwner = window;
            dialog.lpstrFilter = L"CSV files\0*.csv\0All files\0*.*\0";
            dialog.lpstrFile = filename;
            dialog.nMaxFile = MAX_PATH;
            dialog.lpstrDefExt = L"csv";
            dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
            if (GetSaveFileNameW(&dialog))
                try {
                    sim->exportDetector(filename);
                } catch (const std::exception &e) {
                    MessageBoxA(window, e.what(), "Detector export", MB_ICONERROR);
                }
        }
    }
    bool inCanvas(int x, int y) const {
        return x > panelWidth && x < width && y >= 0 && y < height - 64;
    }
    V3 ray(int x, int y) const {
        float w = static_cast<float>(std::max(1, width - panelWidth)),
              h = static_cast<float>(std::max(1, height - 64));
        return camera.ray(2 * (x - panelWidth) / w - 1, 1 - 2 * y / h, w / h);
    }
    void paint(int x, int y) {
        if (!sim)
            return;
        auto d = ray(x, y), e = camera.eye();
        int axis = sim->settings.sliceAxis;
        float a = axis == 0 ? e.x : (axis == 1 ? e.y : e.z), b = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
        float t = std::abs(b) > 1e-6f ? (sim->settings.slice - a) / b : -1;
        auto p = e + d * t;
        float u = static_cast<float>(x - panelWidth) / std::max(1, width - panelWidth);
        float v = 1 - static_cast<float>(y) / std::max(1, height - 64);
        bool overDetector = u > .73f && u < .98f && v > .025f && v < .31f;
        sim->settings.brushActive = inCanvas(x, y) && !overDetector && sim->settings.display != 4 && t > 0 &&
                                    p.x > .02f && p.x < .98f && p.y > .02f && p.y < .98f && p.z > .02f &&
                                    p.z < .98f;
        if (sim->settings.brushActive)
            sim->settings.brush = p;
    }
    void mouse(int x, int y) {
        if (orbit) {
            camera.yaw += (x - previous.x) * .007f;
            camera.pitch = std::clamp(camera.pitch + (y - previous.y) * .007f, -1.4f, 1.4f);
        }
        if (dragging)
            paint(x, y);
        previous = {x, y};
    }
    void stop() {
        dragging = false;
        orbit = false;
        if (sim)
            sim->settings.brushActive = false;
        if (GetCapture() == window)
            ReleaseCapture();
    }
    void draw() {
        gl.BindFramebuffer(0x8D40, 0);
        glViewport(0, 0, width, height);
        glClearColor(.012f, .018f, .032f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_SCISSOR_TEST);
        glScissor(0, 0, panelWidth, height);
        glClearColor(246.f / 255, 248.f / 255, 252.f / 255, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        sim->draw(camera, std::max(1, width - panelWidth), std::max(1, height - 64));
    }
    void updateStatus() {
        sim->measure();
        if (!sim->healthy()) {
            paused = true;
            SetWindowTextW(pause, L"Resume");
        }
        auto text =
            std::format(L"{} cubed / {} cells | GPU {:.2f} ms | {:.0f} fps | time {:.3f} | CFL 0.45\nMin "
                        L"wavelength {:.1f} cells | max |u| {:.3f} | detector peak {:.5f}{}",
                        sim->settings.n, sim->count(), sim->gpuMs, fps, sim->time, sim->cellsPerWavelength(),
                        sim->metrics.bounds[0], sim->metrics.bounds[2],
                        sim->healthy() ? L"" : L" | INVALID: reset required");
        SetWindowTextW(status, text.c_str());
    }
};

LRESULT CALLBACK windowProc(HWND hwnd, UINT message, WPARAM w, LPARAM l) {
    auto app = reinterpret_cast<App *>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (message == WM_NCCREATE) {
        app = static_cast<App *>(reinterpret_cast<CREATESTRUCTW *>(l)->lpCreateParams);
        app->window = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
    }
    if (app) {
        switch (message) {
        case WM_CLOSE:
            app->running = false;
            return 0;
        case WM_SIZE:
            app->resize();
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_GETMINMAXINFO:
            reinterpret_cast<MINMAXINFO *>(l)->ptMinTrackSize = {1080, 760};
            return 0;
        case WM_COMMAND:
            app->command(LOWORD(w), HIWORD(w));
            return 0;
        case WM_HSCROLL:
            app->updateSliders();
            return 0;
        case WM_LBUTTONDOWN:
            if (app->inCanvas(GET_X_LPARAM(l), GET_Y_LPARAM(l))) {
                app->dragging = true;
                SetCapture(hwnd);
                app->paint(GET_X_LPARAM(l), GET_Y_LPARAM(l));
            }
            return 0;
        case WM_RBUTTONDOWN:
            if (app->inCanvas(GET_X_LPARAM(l), GET_Y_LPARAM(l))) {
                app->orbit = true;
                app->previous = {GET_X_LPARAM(l), GET_Y_LPARAM(l)};
                SetCapture(hwnd);
            }
            return 0;
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_CAPTURECHANGED:
            app->stop();
            return 0;
        case WM_MOUSEMOVE:
            app->mouse(GET_X_LPARAM(l), GET_Y_LPARAM(l));
            return 0;
        case WM_MOUSEWHEEL:
            app->camera.distance =
                std::clamp(app->camera.distance * std::exp(-GET_WHEEL_DELTA_WPARAM(w) * .001f), 1.25f, 4.5f);
            return 0;
        case WM_CTLCOLORSTATIC:
            SetTextColor(reinterpret_cast<HDC>(w), RGB(30, 40, 55));
            SetBkColor(reinterpret_cast<HDC>(w), RGB(240, 242, 246));
            return reinterpret_cast<LRESULT>(GetSysColorBrush(COLOR_BTNFACE));
        }
    }
    return DefWindowProcW(hwnd, message, w, l);
}

bool check(Simulation &sim, std::ostream &out, const std::string &label) {
    sim.measure();
    bool ok = sim.healthy();
    out << label << ": " << (ok ? "PASS" : "FAIL") << " energy=" << sim.metrics.totals[0]
        << " maxU=" << sim.metrics.bounds[0] << " detector=" << sim.metrics.bounds[2]
        << " downstream=" << sim.metrics.spatial[0] << " invalid=" << sim.metrics.totals[2]
        << " barrierLeak=" << sim.metrics.totals[3] << " gpuMs=" << sim.gpuMs << "\n";
    return ok;
}

bool gpuTests(App &app, const std::filesystem::path &folder) {
    auto &sim = *app.sim;
    std::ofstream out(folder / "WaveOptics3D-tests.txt");
    out << std::unitbuf << app.gl.driver << "\n";
    bool ok = true;
    auto require = [&](bool pass, const std::string &label) {
        out << label << ": " << (pass ? "PASS" : "FAIL") << "\n";
        ok = ok && pass;
    };
    auto advance = [&](int steps) {
        while (steps > 0) {
            MSG msg{};
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (!app.running)
                throw std::runtime_error("GPU validation cancelled.");
            int batch = std::min(steps, 16);
            sim.tick(batch);
            steps -= batch;
            sim.measure();
            if (!sim.healthy()) {
                ok = false;
                out << "Invalid wave state at t=" << sim.time << "\n";
                break;
            }
        }
    };
    auto screenshot = [&](const std::string &name) {
        app.draw();
        capture(folder / name, panelWidth, 64, app.width - panelWidth, app.height - 64);
        SwapBuffers(app.gl.dc);
    };
    sim.settings = Settings{};
    sim.settings.n = 32;
    sim.settings.source = 2;
    sim.reset();
    advance(64);
    require(sim.metrics.bounds[0] == 0 && sim.metrics.totals[0] == 0 && sim.metrics.bounds[2] == 0,
            "Zero field remains exactly zero");
    // Independent CPU stencil oracle, used only for validation, never interactive physics.
    sim.settings.fixture = 2;
    sim.settings.absorbing = true;
    sim.reset();
    auto u = sim.snapshot(), prev = sim.previousSnapshot(), medium = sim.mediumSnapshot();
    std::vector<float> next(u.size()), average(u.size(), 0);
    int n = sim.settings.n;
    float dt = sim.stepSize(), alpha = 1 - std::exp(-dt / (4 * sim.settings.wavelength));
    for (int step = 0; step < 19; ++step) {
        for (int z = 0; z < n; ++z)
            for (int y = 0; y < n; ++y)
                for (int x = 0; x < n; ++x) {
                    int i = x + n * (y + n * z);
                    float value = 0;
                    if (x > 0 && x < n - 1 && y > 0 && y < n - 1 && z > 0 && z < n - 1) {
                        float lap = u[i - 1] + u[i + 1] + u[i - n] + u[i + n] + u[i - n * n] + u[i + n * n] -
                                    6 * u[i];
                        float damping = medium[2 * i + 1] * dt;
                        value = (2 * u[i] - (1 - damping) * prev[i] + .45f * .45f * lap / medium[2 * i]) /
                                (1 + damping);
                    }
                    next[i] = value;
                    average[i] = (1 - alpha) * average[i] + alpha * value * value;
                }
        prev.swap(u);
        u.swap(next);
    }
    advance(19);
    auto gpu = sim.snapshot(), gpuAverage = sim.intensitySnapshot();
    float error = 0, averageError = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        error = std::max(error, std::abs(u[i] - gpu[i]));
        averageError = std::max(averageError, std::abs(average[i] - gpuAverage[i]));
    }
    out << "CPU/GPU max field error=" << error << " EMA error=" << averageError << "\n";
    require(error < 4e-6f && averageError < 2e-6f, "3D seven-point stencil, ping-pong and intensity oracle");
    double coarsePhaseError = 0;
    for (int grid : {32, 64}) {
        sim.settings = Settings{};
        sim.settings.n = grid;
        sim.settings.fixture = 1;
        sim.settings.absorbing = false;
        sim.settings.source = 2;
        sim.reset();
        double energy0 = sim.metrics.totals[0];
        advance(256);
        gpu = sim.snapshot();
        double sum = 0, reference = 0, pi = std::numbers::pi, s2 = 0;
        for (int mode : {1, 2, 3})
            s2 += std::pow(std::sin(mode * pi / (2 * (grid - 1))), 2);
        double theta = 2 * std::asin(.45 * std::sqrt(s2)), amplitude = std::cos(256 * theta);
        for (int z = 0; z < grid; ++z)
            for (int y = 0; y < grid; ++y)
                for (int x = 0; x < grid; ++x) {
                    double eigen = std::sin(pi * x / (grid - 1)) * std::sin(2 * pi * y / (grid - 1)) *
                                   std::sin(3 * pi * z / (grid - 1));
                    double diff = gpu[x + grid * (y + grid * z)] - amplitude * eigen;
                    sum += diff * diff;
                    reference += eigen * eigen;
                }
        double l2 = std::sqrt(sum / reference), energyError = std::abs(sim.metrics.totals[0] / energy0 - 1);
        double phaseError = std::abs(theta / sim.stepSize() - pi * std::sqrt(14.));
        out << "Eigenmode grid " << grid << ": L2=" << l2 << " energyRelative=" << energyError
            << " dispersionError=" << phaseError << "\n";
        require(l2 < .0002 && energyError < .0002,
                "Discrete 3D eigenfrequency and conserved half-step energy");
        if (grid == 32)
            coarsePhaseError = phaseError;
        else
            require(phaseError < coarsePhaseError * .3, "Second-order dispersion convergence");
    }
    double retained[2]{};
    for (int absorbing = 0; absorbing < 2; ++absorbing) {
        sim.settings = Settings{};
        sim.settings.n = 64;
        sim.settings.fixture = 2;
        sim.settings.source = 2;
        sim.settings.absorbing = absorbing != 0;
        sim.reset();
        double energy0 = sim.metrics.totals[0];
        advance(400);
        retained[absorbing] = sim.metrics.totals[0] / energy0;
        ok = check(sim, out, std::format("Pulse boundary {}", absorbing)) && ok;
    }
    out << "Pulse retained energy reflecting=" << retained[0] << " sponge=" << retained[1] << "\n";
    require(std::abs(retained[0] - 1) < .001 && retained[1] < .05, "Absorber attenuates outgoing 3D waves");
    sim.settings = Settings{};
    sim.settings.n = 64;
    sim.settings.scene = 2;
    sim.settings.refractiveIndex = 1;
    sim.settings.source = 1;
    sim.reset();
    advance(120);
    double pulseEnergy = sim.metrics.totals[0];
    advance(480);
    out << "Driven Gaussian pulse energy initial=" << pulseEnergy << " late=" << sim.metrics.totals[0]
        << "\n";
    require(pulseEnergy > .001 && sim.metrics.totals[0] < pulseEnergy * .01,
            "Finite-duration illumination propagates and decays");
    for (int scene = 0; scene < 3; ++scene) {
        sim.settings = Settings{};
        sim.settings.scene = scene;
        sim.settings.opening = scene == 0 ? .076f : (scene == 1 ? .12f : .18f);
        sim.reset();
        advance(800);
        ok = check(sim, out, std::format("Optical experiment {}", scene)) && ok;
        require(sim.metrics.bounds[2] > .0001f && sim.metrics.spatial[0] > .000001f,
                "Transmission reaches the detector");
        require(sim.metrics.bounds[3] > .01f, "Wave propagation outside the central Z slice");
        auto screen = sim.detectorSnapshot();
        float symmetry = 0, peak = *std::max_element(screen.begin(), screen.end());
        int grid = sim.settings.n;
        for (int z = 0; z < grid; ++z)
            for (int y = 0; y < grid; ++y)
                symmetry =
                    std::max(symmetry, std::abs(screen[y + grid * z] - screen[grid - 1 - y + grid * z]));
        out << "Detector mirror asymmetry=" << symmetry / std::max(peak, 1e-12f) << "\n";
        require(symmetry / std::max(peak, 1e-12f) < .005f, "Symmetric aperture / lens detector response");
        screenshot(std::format("WaveOptics3D-scene-{}.bmp", scene));
        sim.settings.display = 4;
        screenshot(std::format("WaveOptics3D-detector-{}.bmp", scene));
        sim.settings.display = 0;
    }
    for (int grid : {64, 128}) {
        sim.settings = Settings{};
        sim.settings.n = grid;
        sim.settings.scene = 2;
        sim.settings.refractiveIndex = 2;
        sim.settings.wavelength = 8.f * 2 / (grid - 1);
        sim.settings.opening = .22f;
        sim.reset();
        advance(400);
        ok = check(sim, out, std::format("Grid {} at minimum local wavelength and index 2", grid)) && ok;
        require(sim.cellsPerWavelength() >= 7.99f, "At least eight cells per local wavelength");
    }
    double focus[2]{};
    for (int lens = 0; lens < 2; ++lens) {
        sim.settings = Settings{};
        sim.settings.n = 96;
        sim.settings.scene = 2;
        sim.settings.opening = .18f;
        sim.settings.refractiveIndex = lens ? 1.4f : 1.f;
        sim.reset();
        advance(800);
        auto screen = sim.detectorSnapshot();
        int samples = 0;
        for (int z = 0; z < 96; ++z)
            for (int y = 0; y < 96; ++y) {
                float dy = y / 95.f - .5f, dz = z / 95.f - .5f;
                if (dy * dy + dz * dz < .06f * .06f) {
                    focus[lens] += screen[y + 96 * z];
                    ++samples;
                }
            }
        focus[lens] /= samples;
    }
    out << "Central detector mean without lens=" << focus[0] << " with lens=" << focus[1]
        << " gain=" << focus[1] / focus[0] << "\n";
    require(focus[1] > focus[0] * 1.1, "Refractive material changes propagation and concentrates the beam");
    sim.settings = Settings{};
    sim.settings.n = 64;
    sim.settings.source = 2;
    sim.settings.scene = 2;
    sim.settings.refractiveIndex = 1;
    sim.reset();
    sim.settings.brushActive = true;
    sim.settings.brush = {.5f, .5f, .7f};
    advance(80);
    require(sim.metrics.bounds[0] > .01f, "Interactive source injects a 3D wave");
    sim.settings.brushActive = false;
    ok = check(sim, out, "Mouse source finite state") && ok;
    sim.settings.sliceAxis = 2;
    sim.settings.slice = .7f;
    auto v = V3{.5f, .5f, .7f} - app.camera.eye();
    float depth = v.dot(app.camera.forward()), tangent = std::tan(.325f);
    float aspect = static_cast<float>(app.width - panelWidth) / (app.height - 64);
    int sx = panelWidth + static_cast<int>((1 + v.dot(app.camera.right()) / (depth * tangent * aspect)) *
                                           .5f * (app.width - panelWidth));
    int sy = static_cast<int>((1 - v.dot(app.camera.up()) / (depth * tangent)) * .5f * (app.height - 64));
    app.paint(sx, sy);
    require(sim.settings.brushActive && (sim.settings.brush - V3{.5f, .5f, .7f}).length() < .005f,
            "Screen ray selects the requested 3D slice");
    app.stop();
    require(!sim.settings.brushActive, "Mouse capture release stops the source");
    sim.settings = Settings{};
    sim.reset();
    advance(600);
    std::array<std::uint64_t, 5> hashes{};
    for (int mode = 0; mode < 5; ++mode) {
        SendMessageW(app.combos[2], CB_SETCURSEL, mode, 0);
        app.command(12, CBN_SELCHANGE);
        app.draw();
        std::vector<unsigned char> pixels(static_cast<size_t>(app.width - panelWidth) * (app.height - 64) *
                                          4);
        glReadPixels(panelWidth, 64, app.width - panelWidth, app.height - 64, GL_RGBA, GL_UNSIGNED_BYTE,
                     pixels.data());
        std::uint64_t hash = 14695981039346656037ull;
        for (auto byte : pixels) {
            hash ^= byte;
            hash *= 1099511628211ull;
        }
        hashes[mode] = hash;
        screenshot(std::format("WaveOptics3D-view-{}.bmp", mode));
    }
    bool distinct = true;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            distinct &= hashes[i] != hashes[j];
    require(distinct, "Five distinct rendered views");
    sim.exportDetector(folder / "Detector.csv");
    std::ifstream csv(folder / "Detector.csv");
    std::string line;
    int lines = 0;
    while (std::getline(csv, line))
        ++lines;
    require(lines == sim.settings.n * sim.settings.n + 3, "Raw detector CSV export");
    app.command(30, 0);
    bool paused = app.paused;
    app.command(31, 0);
    bool reset = sim.time == 0 && sim.metrics.bounds[0] == 0;
    app.command(32, 0);
    require(paused && reset && app.paused && app.single, "Pause/reset/single-step handlers");
    app.advanceOne();
    require(sim.steps == 1 && std::abs(sim.time - sim.stepSize()) < 1e-9 && !app.single,
            "Single step advances exactly one FDTD update");
    auto beforeOff = sim.steps;
    SendMessageW(app.combos[4], CB_SETCURSEL, 2, 0);
    app.command(14, CBN_SELCHANGE);
    require(sim.settings.source == 2 && sim.steps == beforeOff,
            "Source-off preserves the existing wave field");
    SendMessageW(app.combos[4], CB_SETCURSEL, 1, 0);
    app.command(14, CBN_SELCHANGE);
    require(sim.settings.source == 1 && sim.steps == 0, "Pulse selection restarts the experiment");
    SendMessageW(app.combos[0], CB_SETCURSEL, 2, 0);
    app.command(10, CBN_SELCHANGE);
    SendMessageW(app.combos[1], CB_SETCURSEL, 0, 0);
    app.command(11, CBN_SELCHANGE);
    SendMessageW(app.sliders[2], TBM_SETPOS, TRUE, 100);
    app.updateSliders();
    require(sim.settings.n == 64 && sim.settings.scene == 2 && sim.settings.refractiveIndex == 2 &&
                sim.cellsPerWavelength() >= 8,
            "Native scene/grid/material controls and wavelength safety");
    auto errorCode = glGetError();
    require(errorCode == 0, std::format("OpenGL error {}", errorCode));
    out << (ok ? "ALL PASSED" : "FAILED") << "\n";
    return ok;
}
} // namespace optics

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR args, int show) {
    using namespace optics;
    std::wstring command = args;
    bool tests = command.find(L"--gpu-test") != std::wstring::npos,
         smoke = command.find(L"--smoke") != std::wstring::npos;
    auto folder = std::filesystem::current_path();
    try {
        SetProcessDPIAware();
        INITCOMMONCONTROLSEX cc{sizeof(cc), ICC_BAR_CLASSES};
        InitCommonControlsEx(&cc);
        App app;
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
        auto hwnd = CreateWindowExW(0, wc.lpszClassName, L"Wave Optics Lab / 3D GPU / OpenGL 4.3",
                                    WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, 1360,
                                    1000, nullptr, nullptr, instance, &app);
        if (!hwnd)
            throw std::runtime_error("Window creation failed.");
        app.gl.initialize(hwnd);
        app.sim = std::make_unique<Simulation>(app.gl);
        app.controls();
        modern::apply(hwnd, panelWidth, 286, L"Wave Optics Lab", L"Diffraction and interference");
        app.resize();
        ShowWindow(hwnd, (tests || smoke) ? SW_SHOWNOACTIVATE : show);
        if (tests)
            return gpuTests(app, folder) ? 0 : 1;
        using Clock = std::chrono::steady_clock;
        auto last = Clock::now(), report = last;
        double elapsed = 0;
        int frames = 0, counter = 0;
        while (app.running) {
            MSG msg{};
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT)
                    app.running = false;
                if (msg.message == WM_KEYDOWN && msg.hwnd == hwnd) {
                    if (msg.wParam == VK_ESCAPE)
                        app.running = false;
                    if (msg.wParam == VK_SPACE) {
                        app.toggle();
                        continue;
                    }
                    if (msg.wParam == 'R') {
                        app.sim->reset();
                        continue;
                    }
                    if (msg.wParam == 'N') {
                        app.command(32, 0);
                        continue;
                    }
                }
                if (IsDialogMessageW(hwnd, &msg))
                    continue;
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if (!app.running)
                break;
            auto now = Clock::now();
            elapsed = std::min(.05, elapsed + std::chrono::duration<double>(now - last).count());
            last = now;
            if (IsIconic(hwnd)) {
                Sleep(30);
                elapsed = 0;
                continue;
            }
            if (smoke)
                app.sim->tick();
            else if (app.single) {
                app.advanceOne();
                elapsed = 0;
            } else if (!app.paused && elapsed >= 1. / 60) {
                app.sim->tick();
                elapsed = 0;
            } else if (app.paused)
                elapsed = 0;
            app.draw();
            ++frames;
            ++counter;
            if (smoke && frames == 180) {
                std::ofstream out(folder / "Lab04_smoke.txt");
                out << app.gl.driver << "\n";
                bool ok = check(*app.sim, out, "3D wave optics smoke");
                capture(folder / "Lab04.bmp", panelWidth, 64, app.width - panelWidth, app.height - 64);
                auto error = glGetError();
                out << "OpenGL error: " << error << "\n";
                return ok && error == 0 ? 0 : 1;
            }
            SwapBuffers(app.gl.dc);
            if (now - report > std::chrono::milliseconds(500)) {
                app.fps = counter / static_cast<float>(std::chrono::duration<double>(now - report).count());
                counter = 0;
                report = now;
                app.updateStatus();
            }
            Sleep(1);
        }
        return 0;
    } catch (const std::exception &error) {
        if (tests || smoke) {
            std::ofstream out(folder / "WaveOptics3D-error.txt");
            out << error.what();
        } else
            MessageBoxA(nullptr, error.what(), "Wave Optics Lab 3D", MB_ICONERROR);
        return 1;
    }
}
