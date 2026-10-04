"""MathSystem: vector, matrix, geometry, quaternion and interpolation math."""

from . import acos, asin, atan2, cos, fsum, sin, sqrt

_NUMERIC_TYPES = (int, float)
_EPSILON = 1e-12
_STRASSEN_CUTOFF = 64


def _is_number(value):
    return isinstance(value, _NUMERIC_TYPES) and not isinstance(value, bool)


def _as_vector(value, name):
    if value is None:
        raise TypeError(f"{name} must be a sequence of numbers, got None")
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence of numbers, got {type(value).__name__}"
        )
    try:
        items = list(value)
    except TypeError:
        raise TypeError(
            f"{name} must be a sequence of numbers, got {type(value).__name__}"
        ) from None
    result = []
    for item in items:
        if not _is_number(item):
            raise TypeError(
                f"{name} must contain only numbers, got {type(item).__name__}"
            )
        result.append(float(item))
    return result


def _as_matrix(value, name):
    if value is None or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a matrix (sequence of numeric rows)")
    try:
        raw_rows = list(value)
    except TypeError:
        raise TypeError(f"{name} must be a matrix (sequence of numeric rows)") from None
    rows = [_as_vector(row, f"{name} row {i}") for i, row in enumerate(raw_rows)]
    if rows:
        width = len(rows[0])
        for index, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"{name} row {index} is not rectangular")
    return rows


def _sub(u, v):
    return [x - y for x, y in zip(u, v)]


def _dot3(u, v):
    return fsum(x * y for x, y in zip(u, v))


def _length(v):
    return sqrt(_dot3(v, v))


def _distance(u, v):
    return _length(_sub(u, v))


# ---------------------------------------------------------------------------
# matrix helpers
# ---------------------------------------------------------------------------


def _matmul_transposed(a, b):
    # Transpose b so the inner loop walks contiguous rows instead of
    # strided columns: correct row-by-column math, cache friendly.
    bt = [list(column) for column in zip(*b)]
    return [[_dot3(row, col) for col in bt] for row in a]


def _mat_add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _mat_sub(a, b):
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _pad_square(matrix, size):
    padded = [row + [0.0] * (size - len(row)) for row in matrix]
    padded.extend([[0.0] * size for _ in range(size - len(matrix))])
    return padded


def _strassen(a, b):
    n = len(a)
    if n <= _STRASSEN_CUTOFF:
        return _matmul_transposed(a, b)
    half = n // 2
    a11 = [row[:half] for row in a[:half]]
    a12 = [row[half:] for row in a[:half]]
    a21 = [row[:half] for row in a[half:]]
    a22 = [row[half:] for row in a[half:]]
    b11 = [row[:half] for row in b[:half]]
    b12 = [row[half:] for row in b[:half]]
    b21 = [row[:half] for row in b[half:]]
    b22 = [row[half:] for row in b[half:]]

    m1 = _strassen(_mat_add(a11, a22), _mat_add(b11, b22))
    m2 = _strassen(_mat_add(a21, a22), b11)
    m3 = _strassen(a11, _mat_sub(b12, b22))
    m4 = _strassen(a22, _mat_sub(b21, b11))
    m5 = _strassen(_mat_add(a11, a12), b22)
    m6 = _strassen(_mat_sub(a21, a11), _mat_add(b11, b12))
    m7 = _strassen(_mat_sub(a12, a22), _mat_add(b21, b22))

    c11 = _mat_add(_mat_sub(_mat_add(m1, m4), m5), m7)
    c12 = _mat_add(m3, m5)
    c21 = _mat_add(m2, m4)
    c22 = _mat_add(_mat_add(_mat_sub(m1, m2), m3), m6)
    top = [left + right for left, right in zip(c11, c12)]
    bottom = [left + right for left, right in zip(c21, c22)]
    return top + bottom


def _strassen_multiply(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    size = 1
    while size < max(rows, inner, cols):
        size *= 2
    product = _strassen(_pad_square(a, size), _pad_square(b, size))
    return [row[:cols] for row in product[:rows]]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _normalize_shape(shape):
    if not isinstance(shape, dict) or "type" not in shape:
        raise TypeError("shape must be a dict with a 'type' key")
    kind = str(shape["type"]).lower()
    if kind == "aabb":
        return {
            "type": "aabb",
            "min": _as_vector(shape["min"], "aabb min"),
            "max": _as_vector(shape["max"], "aabb max"),
        }
    if kind == "sphere":
        return {
            "type": "sphere",
            "center": _as_vector(shape["center"], "sphere center"),
            "radius": float(shape["radius"]),
        }
    if kind == "capsule":
        return {
            "type": "capsule",
            "a": _as_vector(shape["a"], "capsule a"),
            "b": _as_vector(shape["b"], "capsule b"),
            "radius": float(shape["radius"]),
        }
    raise ValueError(f"unsupported shape type: {kind}")


def _bounding_sphere(shape):
    if shape["type"] == "sphere":
        return shape["center"], shape["radius"]
    if shape["type"] == "aabb":
        center = [(lo + hi) / 2.0 for lo, hi in zip(shape["min"], shape["max"])]
        return center, _distance(center, shape["max"])
    center = [(x + y) / 2.0 for x, y in zip(shape["a"], shape["b"])]
    radius = _distance(shape["a"], shape["b"]) / 2.0 + shape["radius"]
    return center, radius


def _aabb_closest_point(box, point):
    return [
        max(box["min"][i], min(point[i], box["max"][i])) for i in range(3)
    ]


def _point_segment_distance(point, start, end):
    direction = _sub(end, start)
    denom = _dot3(direction, direction)
    if denom <= _EPSILON:
        ratio = 0.0
    else:
        ratio = _dot3(_sub(point, start), direction) / denom
        ratio = max(0.0, min(1.0, ratio))
    closest = [start[i] + ratio * direction[i] for i in range(3)]
    return _distance(point, closest)


def _segment_segment_distance(p1, q1, p2, q2):
    d1 = _sub(q1, p1)
    d2 = _sub(q2, p2)
    r = _sub(p1, p2)
    a = _dot3(d1, d1)
    e = _dot3(d2, d2)
    f = _dot3(d2, r)
    if a <= _EPSILON and e <= _EPSILON:
        s, t = 0.0, 0.0
    elif a <= _EPSILON:
        s, t = 0.0, max(0.0, min(1.0, f / e))
    elif e <= _EPSILON:
        t = 0.0
        s = max(0.0, min(1.0, -_dot3(d1, r) / a))
    else:
        b = _dot3(d1, d2)
        c = _dot3(d1, r)
        denom = a * e - b * b
        s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom > _EPSILON else 0.0
        t = (b * s + f) / e
        if t < 0.0:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        elif t > 1.0:
            t = 1.0
            s = max(0.0, min(1.0, (b - c) / a))
    c1 = [p1[i] + d1[i] * s for i in range(3)]
    c2 = [p2[i] + d2[i] * t for i in range(3)]
    return _distance(c1, c2)


def _segment_intersects_aabb(start, end, bmin, bmax):
    t_min, t_max = 0.0, 1.0
    for i in range(3):
        delta = end[i] - start[i]
        if abs(delta) < _EPSILON:
            if start[i] < bmin[i] or start[i] > bmax[i]:
                return False
            continue
        t1 = (bmin[i] - start[i]) / delta
        t2 = (bmax[i] - start[i]) / delta
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def _intersect(a, b):
    pair = (a["type"], b["type"])
    if pair == ("sphere", "sphere"):
        return _distance(a["center"], b["center"]) <= a["radius"] + b["radius"] + _EPSILON
    if pair == ("aabb", "aabb"):
        for i in range(3):  # early exit on the first separated axis
            if a["max"][i] < b["min"][i] or b["max"][i] < a["min"][i]:
                return False
        return True
    if pair in (("sphere", "aabb"), ("aabb", "sphere")):
        sphere, box = (a, b) if pair[0] == "sphere" else (b, a)
        closest = _aabb_closest_point(box, sphere["center"])
        return _distance(closest, sphere["center"]) <= sphere["radius"] + _EPSILON
    if pair in (("capsule", "sphere"), ("sphere", "capsule")):
        capsule, sphere = (a, b) if pair[0] == "capsule" else (b, a)
        distance = _point_segment_distance(
            sphere["center"], capsule["a"], capsule["b"]
        )
        return distance <= capsule["radius"] + sphere["radius"] + _EPSILON
    if pair == ("capsule", "capsule"):
        distance = _segment_segment_distance(
            a["a"], a["b"], b["a"], b["b"]
        )
        return distance <= a["radius"] + b["radius"] + _EPSILON
    if pair in (("capsule", "aabb"), ("aabb", "capsule")):
        capsule, box = (a, b) if pair[0] == "capsule" else (b, a)
        radius = capsule["radius"]
        bmin = [v - radius for v in box["min"]]
        bmax = [v + radius for v in box["max"]]
        return _segment_intersects_aabb(capsule["a"], capsule["b"], bmin, bmax)
    raise ValueError(f"unsupported shape pair: {pair}")


def _translate_shape(shape, velocity, time):
    def moved(point):
        return [x + v * time for x, v in zip(point, velocity)]

    if shape["type"] == "sphere":
        return {**shape, "center": moved(shape["center"])}
    if shape["type"] == "aabb":
        return {**shape, "min": moved(shape["min"]), "max": moved(shape["max"])}
    return {**shape, "a": moved(shape["a"]), "b": moved(shape["b"])}


def _swept_spheres(a, b, velocity, dt):
    if _intersect(a, b):
        return True
    relative = _sub(b["center"], a["center"])
    radius = a["radius"] + b["radius"]
    aa = _dot3(velocity, velocity)
    if aa <= _EPSILON:
        return False
    bb = 2.0 * _dot3(relative, velocity)
    cc = _dot3(relative, relative) - radius * radius
    discriminant = bb * bb - 4.0 * aa * cc
    if discriminant < 0.0:
        return False
    hit_time = (-bb - sqrt(discriminant)) / (2.0 * aa)
    return 0.0 <= hit_time <= dt


def _swept_intersect(a, b, velocity, dt, steps):
    if a["type"] == "sphere" and b["type"] == "sphere":
        return _swept_spheres(a, b, velocity, dt)
    # Broadphase: reject when even the swept bounding spheres cannot meet.
    center_a, radius_a = _bounding_sphere(a)
    center_b, radius_b = _bounding_sphere(b)
    if _distance(center_a, center_b) > radius_a + radius_b + _length(velocity) * dt:
        return False
    steps = max(1, int(steps))
    for index in range(steps + 1):  # conservative advancement, exits on first hit
        time = dt * index / steps
        if _intersect(a, _translate_shape(b, velocity, time)):
            return True
    return False


# ---------------------------------------------------------------------------
# quaternion helpers  (w, x, y, z)
# ---------------------------------------------------------------------------


def _as_quaternion(value, name):
    components = _as_vector(value, name)
    if len(components) != 4:
        raise ValueError(
            f"{name} must have exactly 4 components (w, x, y, z), "
            f"got {len(components)}"
        )
    return tuple(components)


def _normalize_quaternion(q):
    w, x, y, z = q
    norm = sqrt(w * w + x * x + y * y + z * z)
    if norm <= _EPSILON:
        raise ValueError("cannot normalize a zero-length quaternion")
    return (w / norm, x / norm, y / norm, z / norm)


def _hamilton_product(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


# ---------------------------------------------------------------------------
# interpolation helpers
# ---------------------------------------------------------------------------


def _lerp(a, b, t):
    if _is_number(a) and _is_number(b):
        return a + (b - a) * t
    va = _as_vector(a, "a")
    vb = _as_vector(b, "b")
    if len(va) != len(vb):
        raise ValueError(
            f"interpolation dimension mismatch: {len(va)} != {len(vb)}"
        )
    return [x + (y - x) * t for x, y in zip(va, vb)]


def _slerp(a, b, t):
    va = _as_vector(a, "a")
    vb = _as_vector(b, "b")
    if len(va) != len(vb):
        raise ValueError(
            f"slerp dimension mismatch: {len(va)} != {len(vb)}"
        )
    norm_a = _length(va)
    norm_b = _length(vb)
    if norm_a <= _EPSILON or norm_b <= _EPSILON:
        return _lerp(va, vb, t)
    unit_a = [x / norm_a for x in va]
    unit_b = [x / norm_b for x in vb]
    dot = max(-1.0, min(1.0, _dot3(unit_a, unit_b)))
    if len(va) == 4 and dot < 0.0:
        # quaternion: negate to take the shortest path around the sphere
        unit_b = [-x for x in unit_b]
        norm_b = -norm_b
        dot = -dot
    if dot > 1.0 - 1e-9:
        direction = _lerp(unit_a, unit_b, t)
    else:
        theta = acos(dot)
        sin_theta = sin(theta)
        scale_a = sin((1.0 - t) * theta) / sin_theta
        scale_b = sin(t * theta) / sin_theta
        direction = [scale_a * x + scale_b * y for x, y in zip(unit_a, unit_b)]
    magnitude = norm_a + (norm_b - norm_a) * t
    return [magnitude * x for x in direction]


def _catmull_rom_scalar(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _catmull_rom(p0, p1, p2, p3, t):
    if _is_number(p1):
        return _catmull_rom_scalar(p0, p1, p2, p3, t)
    return [
        _catmull_rom_scalar(a, b, c, d, t)
        for a, b, c, d in zip(p0, p1, p2, p3)
    ]


def _spline(points, t):
    samples = [
        point if _is_number(point) else _as_vector(point, "spline point")
        for point in points
    ]
    count = len(samples)
    if count == 1:
        return samples[0]
    scaled = t * (count - 1)
    segment = min(int(scaled), count - 2)
    local = scaled - segment
    p0 = samples[max(segment - 1, 0)]
    p1 = samples[segment]
    p2 = samples[segment + 1]
    p3 = samples[min(segment + 2, count - 1)]
    return _catmull_rom(p0, p1, p2, p3, local)


class MathSystem:
    def __init__(self):
        self.state = {}

    def vector_dot(self, a=None, b=None):
        """Dot product with dimension check, type check and precise summation."""
        va = _as_vector(a if a is not None else (), "a")
        vb = _as_vector(b if b is not None else (), "b")
        if len(va) != len(vb):
            raise ValueError(
                f"vector dimension mismatch: {len(va)} != {len(vb)}"
            )
        # fsum is a compensated (error-free) summation, so floating point
        # rounding error does not accumulate across components.
        return fsum(x * y for x, y in zip(va, vb))

    def matrix_multiply(self, a=None, b=None):
        """Matrix product with dimension check, transposed inner loop and
        Strassen acceleration for large matrices."""
        ma = _as_matrix(a if a is not None else [[1.0]], "a")
        mb = _as_matrix(b if b is not None else [[1.0]], "b")
        if not ma or not mb or not ma[0] or not mb[0]:
            raise ValueError("cannot multiply empty matrices")
        if len(ma[0]) != len(mb):
            raise ValueError(
                f"matrix dimension mismatch: {len(ma)}x{len(ma[0])} cannot "
                f"be multiplied by {len(mb)}x{len(mb[0])}"
            )
        if min(len(ma), len(ma[0]), len(mb[0])) >= _STRASSEN_CUTOFF:
            return _strassen_multiply(ma, mb)
        return _matmul_transposed(ma, mb)

    def check_intersection(self, shape_a=None, shape_b=None,
                           vel_a=None, vel_b=None, dt=1.0, steps=16):
        """Intersection test for AABB / sphere / capsule shapes.

        Pass ``vel_a`` / ``vel_b`` (units per ``dt``) for continuous
        (swept) detection so fast objects cannot tunnel through each other.
        Broadphase bounding spheres provide an early exit before the
        narrow phase runs.
        """
        if shape_a is None or shape_b is None:
            return False
        a = _normalize_shape(shape_a)
        b = _normalize_shape(shape_b)
        if vel_a is None and vel_b is None:
            center_a, radius_a = _bounding_sphere(a)
            center_b, radius_b = _bounding_sphere(b)
            if _distance(center_a, center_b) > radius_a + radius_b + _EPSILON:
                return False  # broadphase early exit
            return _intersect(a, b)
        va = _as_vector(vel_a, "vel_a") if vel_a is not None else [0.0, 0.0, 0.0]
        vb = _as_vector(vel_b, "vel_b") if vel_b is not None else [0.0, 0.0, 0.0]
        relative_velocity = _sub(vb, va)
        return _swept_intersect(a, b, relative_velocity, float(dt), steps)

    def quaternion_multiply(self, q1=None, q2=None):
        """Hamilton product of two quaternions (w, x, y, z).

        Both operands and the result are normalized, so chained rotations
        keep unit length instead of drifting.
        """
        qa = _normalize_quaternion(
            _as_quaternion(q1 if q1 is not None else (1.0, 0.0, 0.0, 0.0), "q1")
        )
        qb = _normalize_quaternion(
            _as_quaternion(q2 if q2 is not None else (1.0, 0.0, 0.0, 0.0), "q2")
        )
        return _normalize_quaternion(_hamilton_product(qa, qb))

    def quaternion_inverse(self, q):
        """Inverse rotation: conjugate of the normalized quaternion."""
        w, x, y, z = _normalize_quaternion(_as_quaternion(q, "q"))
        return (w, -x, -y, -z)

    def quaternion_from_euler(self, roll, pitch, yaw):
        """Build a normalized quaternion from euler angles (radians)."""
        half_roll, half_pitch, half_yaw = roll / 2.0, pitch / 2.0, yaw / 2.0
        cr, sr = cos(half_roll), sin(half_roll)
        cp, sp = cos(half_pitch), sin(half_pitch)
        cy, sy = cos(half_yaw), sin(half_yaw)
        return _normalize_quaternion((
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ))

    def quaternion_to_euler(self, q):
        """Convert a quaternion back to (roll, pitch, yaw) in radians.

        The pitch term is clamped to [-1, 1] before asin so the conversion
        stays stable at the gimbal-lock singularities.
        """
        w, x, y, z = _normalize_quaternion(_as_quaternion(q, "q"))
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = atan2(sinr_cosp, cosr_cosp)
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = asin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = atan2(siny_cosp, cosy_cosp)
        return (roll, pitch, yaw)

    def interpolate(self, a=None, b=None, t=0.0, method="linear",
                    duration=None, time=None, loop=False, points=None):
        """Interpolate between values.

        method: "linear" (lerp), "spherical"/"slerp", or "spline"
        (Catmull-Rom through ``points``). ``duration`` + ``time`` give
        duration-independent time scaling; ``loop`` wraps t so the
        interpolation returns to the start instead of stopping.
        """
        if duration is not None:
            if duration <= 0:
                raise ValueError("duration must be positive")
            t = (0.0 if time is None else float(time)) / float(duration)
        t = float(t)
        if loop:
            t = t % 1.0  # wrap around: the end leads back to the start
        else:
            t = max(0.0, min(1.0, t))
        kind = str(method).lower()
        if kind == "spline":
            control = list(points) if points is not None else [
                p for p in (a, b) if p is not None
            ]
            if not control:
                return 0.0
            return _spline(control, t)
        if a is None or b is None:
            if a is None and b is None:
                return 0.0
            return a if b is None else b
        if kind == "linear":
            return _lerp(a, b, t)
        if kind in ("spherical", "slerp"):
            return _slerp(a, b, t)
        raise ValueError(f"unknown interpolation method: {method}")
