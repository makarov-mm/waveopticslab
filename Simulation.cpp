#include "Lab.hpp"
#include <format>
namespace physics {
namespace {
class Optics final : public Lab {
    static constexpr int n = 192;
    std::vector<float> u, old, next, intensity;
    float phase{}, screenPeak{};

  public:
    Optics() {
        reset();
    }
    Description describe() const override {
        return {L"04 / Wave Optics Lab",
                L"Scalar wave propagation through one or two adjustable apertures.",
                {L"Double slit / wave", L"Single slit / wave", L"Double slit / intensity"},
                {L"Wavelength", L"Slit separation"},
                L"Left hold: local wave source\nWarm colors: positive amplitude\nCool colors: negative "
                L"amplitude\nRight edge: detector intensity."};
    }
    void reset() override {
        u.assign(n * n, 0);
        old = u;
        next = u;
        intensity = u;
        phase = 0;
        screenPeak = 0;
    }
    void step(float, const Input &in) override {
        constexpr float courant = .45f;
        for (int k = 0; k < 5; ++k) {
            phase += 2 * pi * courant / (7 + parameters[0] * 20);
            int gap = static_cast<int>(10 + parameters[1] * 45);
            for (int y = 1; y < n - 1; ++y)
                for (int x = 1; x < n - 1; ++x) {
                    int i = y * n + x;
                    float lap = u[i - 1] + u[i + 1] + u[i - n] + u[i + n] - 4 * u[i];
                    int edge = std::min({x, y, n - 1 - x, n - 1 - y});
                    float damping = edge < 18 ? .12f * std::pow(1 - edge / 18.f, 2.f) : .0002f;
                    next[i] = (2 * u[i] - (1 - damping) * old[i] + courant * courant * lap) / (1 + damping);
                    bool aperture = mode == 1
                                        ? std::abs(y - n / 2) < 5
                                        : (std::abs(y - n / 2 - gap) < 4 || std::abs(y - n / 2 + gap) < 4);
                    if (x == n / 3 && !aperture)
                        next[i] = 0;
                    if (x == 20)
                        next[i] += .06f * std::sin(phase) * std::pow(std::sin(pi * y / (n - 1)), 2.f);
                    if (in.left && std::abs(x - static_cast<int>(in.mouse.x * n)) < 2 &&
                        std::abs(y - static_cast<int>(in.mouse.y * n)) < 2)
                        next[i] += .1f * std::sin(phase);
                    intensity[i] = intensity[i] * .99f + next[i] * next[i] * .01f;
                }
            old.swap(u);
            u.swap(next);
        }
        screenPeak = 0;
        for (int y = 0; y < n; ++y)
            screenPeak = std::max(screenPeak, intensity[y * n + n - 22]);
    }
    void render(Canvas &c) const override {
        c.field(n, [&](int x, int y) {
            int i = y * n + x;
            int gap = static_cast<int>(10 + parameters[1] * 45);
            bool hole = mode == 1 ? std::abs(y - n / 2) < 5
                                  : (std::abs(y - n / 2 - gap) < 4 || std::abs(y - n / 2 + gap) < 4);
            if (x == n / 3 && !hole)
                return Color{.7f, .75f, .8f};
            if (mode == 2)
                return mix({.01f, .02f, .04f}, {1, .78f, .25f}, std::sqrt(intensity[i]) * 2);
            return u[i] > 0 ? mix({.015f, .025f, .05f}, {1, .47f, .18f}, u[i] * 2)
                            : mix({.015f, .025f, .05f}, {.1f, .65f, 1}, -u[i] * 2);
        });
        for (int y = 1; y < n; ++y) {
            float a = intensity[(y - 1) * n + n - 22] / std::max(screenPeak, 1e-8f),
                  b = intensity[y * n + n - 22] / std::max(screenPeak, 1e-8f);
            c.line({.99f - a * .12f, (y - 1.f) / n}, {.99f - b * .12f, static_cast<float>(y) / n},
                   {.8f, 1, .7f});
        }
    }
    std::wstring status() const override {
        return std::format(L"192 x 192 FDTD | Courant 0.45 | wavelength {:.1f} cells",
                           7 + parameters[0] * 20);
    }
    bool healthy() const override {
        for (float v : u)
            if (!finite(v) || std::abs(v) > 100)
                return false;
        return true;
    }
};
} // namespace
std::unique_ptr<Lab> makeOptics() {
    return std::make_unique<Optics>();
}
} // namespace physics
