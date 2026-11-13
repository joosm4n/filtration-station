#include <array>
#include <cassert>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <span>

template<typename T, std::size_t HEIGHT, std::size_t WIDTH = HEIGHT>
// using nd_array = std::array<std::array<T, M>, N>;
struct nd_array {
    using value_type = T;
    using size_type = std::size_t;
    using row_type = std::array<T, WIDTH>;
    using col_type = std::array<T, HEIGHT>;
    using storage_type = std::array<row_type, HEIGHT>;
    using iterator = T*;
    using const_iterator = const T*;
    using index_type = std::ptrdiff_t;

    storage_type _raw {};
    static_assert(HEIGHT > 0 && WIDTH > 0);
    static_assert(sizeof(storage_type) == HEIGHT * WIDTH * sizeof(T));

    // construction
    constexpr nd_array() = default;

    constexpr explicit nd_array(const T& value) {
        fill(value);
    }

    constexpr nd_array(const storage_type& raw) : _raw(raw) {}

    constexpr nd_array(std::initializer_list<std::initializer_list<T>> rows) {
        assert(rows.size() <= HEIGHT);
        std::size_t i = 0;
        for (const auto& row : rows) {
            assert(row.size() <= WIDTH);
            std::size_t j = 0;
            for (const auto& v : row)
                _raw[i][j++] = v;
            ++i;
        }
    }

    // shape

    template<typename RT = size_type>
    static constexpr RT num_rows() noexcept {
        static_assert(
            static_cast<int64_t>(HEIGHT)
            >= static_cast<int64_t>(std::numeric_limits<RT>::min())
        );
        static_assert(
            static_cast<int64_t>(HEIGHT)
            <= static_cast<int64_t>(std::numeric_limits<RT>::max())
        );
        return static_cast<RT>(HEIGHT);
    }

    template<typename RT>
    static constexpr RT num_cols() noexcept {
        static_assert(
            static_cast<int64_t>(WIDTH)
            >= static_cast<int64_t>(std::numeric_limits<RT>::min())
        );
        static_assert(
            static_cast<int64_t>(WIDTH)
            <= static_cast<int64_t>(std::numeric_limits<RT>::max())
        );
        return static_cast<RT>(WIDTH);
    }

    static constexpr size_type size() noexcept {
        return HEIGHT * WIDTH;
    }

    // row / column ranges
    constexpr row_type& row(size_type i) noexcept {
        return _raw[i];
    }

    constexpr const row_type& row(size_type i) const noexcept {
        return _raw[i];
    }

    constexpr std::span<row_type, HEIGHT> rows() noexcept {
        return _raw;
    }

    constexpr std::span<const row_type, HEIGHT> rows() const noexcept {
        return _raw;
    }

    // one column as a view (columns aren't contiguous, so this is a transform view)
    constexpr auto column(size_type j) noexcept {
        static_assert(j < WIDTH);
        return _raw | std::views::transform([j](row_type& row) -> T& {
                   return row[j];
               });
    }

    constexpr auto column(size_type j) const noexcept {
        static_assert(j < WIDTH);
        return _raw
            | std::views::transform([j](const row_type& row) -> const T& {
                   return row[j];
               });
    }

    // range of column views
    constexpr auto cols() noexcept {
        return std::views::iota(size_type {0}, WIDTH)
            | std::views::transform([this](size_type j) { return column(j); });
    }

    constexpr auto cols() const noexcept {
        return std::views::iota(size_type {0}, WIDTH)
            | std::views::transform([this](size_type j) { return column(j); });
    }

    // element access
    constexpr row_type& operator[](size_type i) noexcept {
        return _raw[i];
    }

    constexpr const row_type& operator[](size_type i) const noexcept {
        return _raw[i];
    }

    constexpr T& operator()(size_type i, size_type j) noexcept {
        return _raw[i][j];
    }

    constexpr const T& operator()(size_type i, size_type j) const noexcept {
        return _raw[i][j];
    }

    constexpr T& at(size_type i, size_type j) {
        return _raw.at(i).at(j);
    }

    constexpr const T& at(size_type i, size_type j) const {
        return _raw.at(i).at(j);
    }

    constexpr T* data() noexcept {
        return _raw.front().data();
    }

    constexpr const T* data() const noexcept {
        return _raw.front().data();
    }

    // iteration (over rows)
    constexpr iterator begin() noexcept {
        return data();
    }

    constexpr iterator end() noexcept {
        return data() + size();
    }

    constexpr const_iterator begin() const noexcept {
        return data();
    }

    constexpr const_iterator end() const noexcept {
        return data() + size();
    }

    constexpr const_iterator cbegin() const noexcept {
        return begin();
    }

    constexpr const_iterator cend() const noexcept {
        return end();
    }

    constexpr auto span() noexcept {
        return std::span(data(), size());
    }

    constexpr const auto span() const noexcept {
        return std::span(data(), size());
    }

    // modifiers
    constexpr void fill(const T& value) {
        for (auto& row : _raw)
            row.fill(value);
    }

    template<typename F>
    constexpr void mut_map(F map_func) {
        for (T& v : span()) {
            v = map_func(v);
        }
    }

    template<typename F>
    constexpr nd_array map(F map_func) const {
        nd_array arr {};
        auto dest = arr.span();
        auto src = this->span();
        for (size_type i = 0; i < size(); ++i) {
            dest[i] = map_func(src[i]);
        }
        return arr;
    }

    template<typename F>
    constexpr auto filter(F filter_func) noexcept {
        return span() | std::views::filter(filter_func);
    }

    constexpr void
    swap(nd_array& other) noexcept(std::is_nothrow_swappable_v<T>) {
        _raw.swap(other._raw);
    }

    constexpr void swap_row(size_type i, row_type& other) noexcept(
        std::is_nothrow_swappable_v<T>
    ) {
        assert(i < HEIGHT);
        _raw[i].swap(other);
    }

    constexpr void swap_col(size_type j, col_type& other) noexcept(
        std::is_nothrow_swappable_v<T>
    ) {
        assert(j < WIDTH);
        for (size_type i = 0; i < HEIGHT; ++i)
            std::swap(_raw[i][j], other[i]);
    }

    // comparison
    constexpr bool operator==(const nd_array&) const = default;
    constexpr auto operator<=>(const nd_array&) const = default;

    // generic: any shape is a predicate on (row, col)
    template<typename Pred>
        requires std::predicate<Pred&, size_type, size_type>
    constexpr void fill_if(Pred pred, const T& value) {
        for (size_type i = 0; i < HEIGHT; ++i)
            for (size_type j = 0; j < WIDTH; ++j)
                if (pred(i, j))
                    _raw[i][j] = value;
    }

    // rectangle, clipped to the grid
    constexpr void fill_rect(
        size_type top,
        size_type left,
        size_type h,
        size_type w,
        const T& value
    ) {
        const size_type bottom = std::min(top + h, HEIGHT);
        const size_type right = std::min(left + w, WIDTH);
        for (size_type i = top; i < bottom; ++i)
            for (size_type j = left; j < right; ++j)
                _raw[i][j] = value;
    }

    // filled disc centred on (cy, cx), clipped to the grid
    constexpr void
    fill_circle(index_type cy, index_type cx, index_type r, const T& value) {
        constexpr index_type H = static_cast<index_type>(HEIGHT);
        constexpr index_type W = static_cast<index_type>(WIDTH);
        for (index_type i = std::max<index_type>(cy - r, 0);
             i <= std::min(cy + r, H - 1);
             ++i)
            for (index_type j = std::max<index_type>(cx - r, 0);
                 j <= std::min(cx + r, W - 1);
                 ++j) {
                const index_type di = i - cy, dj = j - cx;
                if (di * di + dj * dj <= r * r)
                    _raw[i][j] = value;
            }
    }
};

namespace shape {
constexpr auto circle(long cy, long cx, long r) {
    return [=](std::size_t i, std::size_t j) {
        const long di = long(i) - cy, dj = long(j) - cx;
        return di * di + dj * dj <= r * r;
    };
}

constexpr auto
rect(std::size_t top, std::size_t left, std::size_t h, std::size_t w) {
    // unsigned wraparound makes (i - top < h) equivalent to top <= i < top + h
    return [=](std::size_t i, std::size_t j) {
        return i - top < h && j - left < w;
    };
}

constexpr auto operator_and_not(auto a, auto b) { // a minus b
    return [=](std::size_t i, std::size_t j) { return a(i, j) && !b(i, j); };
}
}; // namespace shape

template<typename T, std::size_t N, std::size_t M>
struct std::formatter<nd_array<T, N, M>> {
    std::formatter<T> elem; // the element formatter handles the spec

    constexpr auto parse(std::format_parse_context& ctx) {
        return elem.parse(ctx);
    }

    auto format(const nd_array<T, N, M>& a, std::format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "[\n");
        for (const auto& row : a.rows()) {
            *out++ = '{';
            for (const auto& v : row) {
                ctx.advance_to(out);
                out = elem.format(v, ctx);
                *out++ = ',';
            }
            out = std::format_to(out, "}},\n");
        }
        *out++ = ']';
        return out;
    }
};

namespace array_misc {
template<typename T, std::size_t N>
constexpr T mean(std::span<T> arr) {
    T sum = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum / arr.size();
}
} // namespace array_misc
