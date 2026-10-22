//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;

/// ECMA-262 #sec-math.random (snapshot e28783d5fc9d) gives each realm's
/// Math.random a distinct stream. Keep the state on that function, so borrowed
/// calls advance the originating stream and independent agents cannot race.
/// RandomState supplies a fresh randomized seed without adding a host API or
/// coupling Math.random to an embedder's clock. This is not Web Crypto.
struct RandomStream(std::cell::Cell<u64>);

impl RandomStream {
    fn new() -> Self {
        use std::hash::{BuildHasher, Hasher};
        let seed = std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish();
        // Xorshift's all-zero state would never advance.
        Self(std::cell::Cell::new(seed.max(1)))
    }

    fn next(&self) -> f64 {
        let mut x = self.0.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0.set(x);
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl crate::value::NativeCallableRetained for RandomStream {
    fn scan_retained_memory(&self, _visitor: &mut dyn crate::value::NativeRetainedMemoryVisitor) {
        // The reporter's inline Cell is counted by the native-callable walker;
        // there are no further allocations.
    }
}

pub(crate) fn nf_math_sqrt(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    if let [Value::Num(x)] = args {
        return Ok(Value::Num(x.sqrt()));
    }
    let x = ab(i.to_number(&arg(args, 0)))?;
    Ok(Value::Num(x.sqrt()))
}

pub(super) fn install_math(it: &mut Interp) {
    let math = it.new_object();
    // The Math constants are { writable:false, enumerable:false, configurable:false }.
    for (name, val) in [
        ("E", std::f64::consts::E),
        ("LN10", std::f64::consts::LN_10),
        ("LN2", std::f64::consts::LN_2),
        ("LOG10E", std::f64::consts::LOG10_E),
        ("LOG2E", std::f64::consts::LOG2_E),
        ("PI", std::f64::consts::PI),
        ("SQRT1_2", std::f64::consts::FRAC_1_SQRT_2),
        ("SQRT2", std::f64::consts::SQRT_2),
    ] {
        math.borrow_mut()
            .props
            .insert(name, Property::data(Value::Num(val), false, false, false));
    }
    // Math[@@toStringTag] = "Math" (non-writable, non-enumerable, configurable).
    if let Some(key) = well_known_key(it, "toStringTag") {
        math.borrow_mut()
            .props
            .insert(key, Property::data(Value::lstr("Math"), false, false, true));
    }
    macro_rules! unary {
        ($name:expr, $f:expr, $intrinsic:expr) => {{
            let f: NativeFn = |i, _t, a| {
                if let [Value::Num(x)] = a {
                    return Ok(Value::Num($f(*x)));
                }
                let x = ab(i.to_number(&arg(a, 0)))?;
                Ok(Value::Num($f(x)))
            };
            register_numeric_only_native(f);
            register_math_intrinsic(f, $intrinsic);
            it.def_method(&math, $name, 1, f);
        }};
        ($name:expr, $f:expr) => {{
            let f: NativeFn = |i, _t, a| {
                // ECMA-262 ToNumber is the identity for Numbers with no observable effects
                // (§7.1.4), so bypass the generic dispatch for the common numeric case.
                // All other inputs retain the complete coercive path with its abrupt
                // completions and valueOf/toString side effects in order.
                if let [Value::Num(x)] = a {
                    return Ok(Value::Num($f(*x)));
                }
                let x = ab(i.to_number(&arg(a, 0)))?;
                Ok(Value::Num($f(x)))
            };
            register_numeric_only_native(f);
            it.def_method(&math, $name, 1, f);
        }};
    }
    unary!("abs", f64::abs, crate::bytecode::INTRINSIC_MATH_ABS);
    unary!("floor", f64::floor, crate::bytecode::INTRINSIC_MATH_FLOOR);
    unary!("ceil", f64::ceil, crate::bytecode::INTRINSIC_MATH_CEIL);
    // Math.round ties toward +Inf and keeps a negative sign for [-0.5, 0). Computed from floor(x)
    // (not floor(x + 0.5), which wrongly rounds up e.g. 0.5 - ε/4 and some large odd integers).
    unary!("round", |x: f64| {
        if x.is_nan() || x.is_infinite() || x == 0.0 {
            x
        } else {
            let f = x.floor();
            let r = if x - f >= 0.5 { f + 1.0 } else { f };
            if r == 0.0 && x < 0.0 {
                -0.0
            } else {
                r
            }
        }
    });
    unary!("trunc", f64::trunc, crate::bytecode::INTRINSIC_MATH_TRUNC);
    it.def_method(&math, "sqrt", 1, nf_math_sqrt);
    unary!("cbrt", f64::cbrt);
    unary!("sign", |x: f64| if x.is_nan() || x == 0.0 {
        x
    } else {
        x.signum()
    });
    unary!("expm1", f64::exp_m1);
    unary!("log1p", f64::ln_1p);
    unary!("sinh", f64::sinh);
    unary!("cosh", f64::cosh);
    unary!("tanh", f64::tanh);
    unary!("asinh", f64::asinh);
    unary!("acosh", f64::acosh);
    // fdlibm atanh — the platform libm loses ~400 ulp near |x| = 1; log1p keeps it exact.
    unary!("atanh", |x: f64| {
        let ax = x.abs();
        if ax > 1.0 {
            return f64::NAN;
        }
        let t = if ax < 0.5 {
            let t = ax + ax;
            0.5 * (t + t * ax / (1.0 - ax)).ln_1p()
        } else {
            0.5 * ((ax + ax) / (1.0 - ax)).ln_1p()
        };
        if x < 0.0 {
            -t
        } else if x == 0.0 {
            x
        } else {
            t
        }
    });
    unary!(
        "fround",
        |x: f64| x as f32 as f64,
        crate::bytecode::INTRINSIC_MATH_FROUND
    );
    unary!(
        "f16round",
        |x: f64| crate::value::f16_to_f32(crate::value::f64_to_f16(x)) as f64
    );
    unary!(
        "clz32",
        |x: f64| (to_uint32(x)).leading_zeros() as f64,
        crate::bytecode::INTRINSIC_MATH_CLZ32
    );
    it.def_method(&math, "sumPrecise", 1, |i, _t, a| {
        // Iterate the argument, requiring every element to be a Number; compute the correctly
        // rounded sum. Infinities dominate (mixed signs → NaN), any NaN → NaN, empty → -0.
        let (iter, next) = ab(i.get_iterator(&arg(a, 0)))?;
        let mut finite: Vec<f64> = Vec::new();
        let (mut pos_inf, mut neg_inf, mut nan) = (false, false, false);
        loop {
            let item = match ab(i.iterator_step(&iter, &next))? {
                Some(v) => v,
                None => break,
            };
            match item {
                Value::Num(n) => {
                    if n.is_nan() {
                        nan = true;
                    } else if n.is_infinite() {
                        if n > 0.0 {
                            pos_inf = true;
                        } else {
                            neg_inf = true;
                        }
                    } else {
                        finite.push(n);
                    }
                }
                _ => {
                    i.iterator_close(&iter);
                    return Err(i.make_error("TypeError", "Math.sumPrecise: not a number"));
                }
            }
        }
        let result = if pos_inf && neg_inf || nan {
            f64::NAN
        } else if pos_inf {
            f64::INFINITY
        } else if neg_inf {
            f64::NEG_INFINITY
        } else if finite.is_empty() {
            -0.0
        } else {
            let s = fsum_exact(&finite);
            if s.is_finite() {
                s
            } else {
                // The exact-summation partials transiently overflowed. Retry on values scaled by a
                // power of two (exact, so the correctly-rounded result is unchanged) centred near
                // 2^500, then scale back — a genuine overflow re-materialises as ±Infinity.
                let max_abs = finite.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                let scale_exp = max_abs.log2().floor() as i32 - 500;
                let down = 2f64.powi(-scale_exp);
                let up = 2f64.powi(scale_exp);
                let scaled: Vec<f64> = finite.iter().map(|&x| x * down).collect();
                fsum_exact(&scaled) * up
            }
        };
        Ok(Value::Num(result))
    });
    it.def_method(&math, "hypot", 2, |i, _t, a| {
        // Coerce every argument (in order), then: any infinite operand yields +Infinity (even
        // alongside a NaN), otherwise any NaN yields NaN, otherwise the Euclidean norm. A pair of
        // Numbers can use the platform's stable two-operand implementation without any coercion
        // work; small numeric calls use the allocation-free common-case helper below
        // (ECMA-262 §21.3.2.19).
        match a {
            [Value::Num(x)] => return Ok(Value::Num(x.abs())),
            [Value::Num(x), Value::Num(y)] => return Ok(Value::Num(x.hypot(*y))),
            [Value::Num(x), Value::Num(y), Value::Num(z)] => {
                return Ok(Value::Num(hypot_converted(&[*x, *y, *z])))
            }
            [Value::Num(x), Value::Num(y), Value::Num(z), Value::Num(w)] => {
                return Ok(Value::Num(hypot_converted(&[*x, *y, *z, *w])))
            }
            _ => {}
        }
        let mut nums = Vec::with_capacity(a.len());
        for v in a {
            nums.push(ab(i.to_number(v))?);
        }
        Ok(Value::Num(hypot_converted(&nums)))
    });
    let imul_fn: NativeFn = |i, _t, a| {
        if let [Value::Num(x), Value::Num(y)] = a {
            let x = to_uint32(*x) as i32;
            let y = to_uint32(*y) as i32;
            return Ok(Value::Num(x.wrapping_mul(y) as f64));
        }
        let x = to_uint32(ab(i.to_number(&arg(a, 0)))?) as i32;
        let y = to_uint32(ab(i.to_number(&arg(a, 1)))?) as i32;
        Ok(Value::Num(x.wrapping_mul(y) as f64))
    };
    // With two Number arguments no coercion runs (ECMA-262 §21.3.2.20 ToUint32 on a Number).
    register_numeric_only_native(imul_fn);
    register_math_intrinsic(imul_fn, crate::bytecode::INTRINSIC_MATH_IMUL);
    it.def_method(&math, "imul", 2, imul_fn);
    let random_stream = Rc::new(RandomStream::new());
    let captured = random_stream.clone();
    let random = it.new_native_fn(
        "random",
        0,
        Rc::new(move |_i, _t, _a| Ok(Value::Num(captured.next()))),
        crate::value::NativeCaptures::Reported(random_stream),
    );
    math.borrow_mut()
        .props
        .insert("random", Property::builtin(random));
    unary!("log", f64::ln);
    unary!("log2", f64::log2);
    unary!("log10", f64::log10);
    unary!("exp", f64::exp);
    unary!("sin", f64::sin);
    unary!("cos", f64::cos);
    unary!("tan", f64::tan);
    unary!("atan", f64::atan);
    unary!("asin", f64::asin);
    unary!("acos", f64::acos);
    it.def_method(&math, "pow", 2, |i, _t, a| {
        if let [Value::Num(base), Value::Num(exp)] = a {
            return Ok(Value::Num(math_pow_numbers(*base, *exp)));
        }
        let base = ab(i.to_number(&arg(a, 0)))?;
        let exp = ab(i.to_number(&arg(a, 1)))?;
        Ok(Value::Num(math_pow_numbers(base, exp)))
    });
    it.def_method(&math, "atan2", 2, |i, _t, a| {
        if let [Value::Num(y), Value::Num(x)] = a {
            return Ok(Value::Num(y.atan2(*x)));
        }
        Ok(Value::Num(
            ab(i.to_number(&arg(a, 0)))?.atan2(ab(i.to_number(&arg(a, 1)))?),
        ))
    });
    let max_fn: NativeFn = |i, _t, a| {
        // Number arguments are already the result of ToNumber, so this branch can reduce them
        // without allocating the spec's intermediate List (ECMA-262 §21.3.2.25). If a non-Number
        // appears, restart through the complete coercion path so every argument is still coerced
        // in order.
        let mut highest = f64::NEG_INFINITY;
        let mut saw_nan = false;
        for v in a {
            let Value::Num(number) = v else {
                return math_max_generic(i, a);
            };
            if number.is_nan() {
                saw_nan = true;
            } else if (*number == 0.0
                && highest == 0.0
                && number.is_sign_positive()
                && highest.is_sign_negative())
                || *number > highest
            {
                highest = *number;
            }
        }
        Ok(Value::Num(if saw_nan { f64::NAN } else { highest }))
    };
    register_numeric_only_native(max_fn);
    register_math_intrinsic(max_fn, crate::bytecode::INTRINSIC_MATH_MAX);
    it.def_method(&math, "max", 2, max_fn);
    let min_fn: NativeFn = |i, _t, a| {
        // See Math.max above: the all-Number case has no observable coercion work to perform
        // (ECMA-262 §21.3.2.26).
        let mut lowest = f64::INFINITY;
        let mut saw_nan = false;
        for v in a {
            let Value::Num(number) = v else {
                return math_min_generic(i, a);
            };
            if number.is_nan() {
                saw_nan = true;
            } else if (*number == 0.0
                && lowest == 0.0
                && number.is_sign_negative()
                && lowest.is_sign_positive())
                || *number < lowest
            {
                lowest = *number;
            }
        }
        Ok(Value::Num(if saw_nan { f64::NAN } else { lowest }))
    };
    register_numeric_only_native(min_fn);
    register_math_intrinsic(min_fn, crate::bytecode::INTRINSIC_MATH_MIN);
    it.def_method(&math, "min", 2, min_fn);
    set_to_string_tag(it, &math, "Math");
    set_builtin(&it.global, "Math", Value::Obj(math));
}

/// Compute Math.hypot after all arguments have completed ToNumber. The ordinary range uses the
/// cheaper direct sum; the scaled retry protects finite values from overflow/underflow while
/// retaining the specification's implementation-approximated result allowance.
fn hypot_converted(values: &[f64]) -> f64 {
    let mut naive_sum = 0.0;
    let mut has_nonzero = false;
    let mut any_inf = false;
    let mut any_nan = false;
    for &n in values {
        if n.is_infinite() {
            any_inf = true;
        } else if n.is_nan() {
            any_nan = true;
        } else {
            has_nonzero |= n != 0.0;
            naive_sum += n * n;
        }
    }
    if any_inf {
        return f64::INFINITY;
    }
    if any_nan {
        return f64::NAN;
    }
    if !has_nonzero {
        return 0.0;
    }
    if naive_sum.is_finite() && naive_sum != 0.0 {
        return naive_sum.sqrt();
    }

    let mut max_abs = 0.0;
    let mut scaled_sum = 0.0;
    for &n in values {
        let abs = n.abs();
        if abs > max_abs {
            scaled_sum = if max_abs == 0.0 {
                1.0
            } else {
                let ratio = max_abs / abs;
                scaled_sum * ratio * ratio + 1.0
            };
            max_abs = abs;
        } else if max_abs != 0.0 {
            let ratio = abs / max_abs;
            scaled_sum += ratio * ratio;
        }
    }
    max_abs * scaled_sum.sqrt()
}

fn math_pow_numbers(base: f64, exp: f64) -> f64 {
    // Number::exponentiate special cases Rust's powf doesn't share: a NaN exponent is NaN even
    // for base 1, and a base of ±1 with an infinite exponent is NaN.
    if exp.is_nan() || (base.abs() == 1.0 && exp.is_infinite()) {
        f64::NAN
    } else {
        base.powf(exp)
    }
}

fn math_max_generic(i: &mut Interp, a: &[Value]) -> Result<Value, Value> {
    // ToNumber every argument first (side effects in order), then reduce. +0 is larger than -0.
    let mut nums = Vec::with_capacity(a.len());
    for v in a {
        nums.push(ab(i.to_number(v))?);
    }
    let mut highest = f64::NEG_INFINITY;
    for n in nums {
        if n.is_nan() {
            return Ok(Value::Num(f64::NAN));
        }
        if n > highest
            || (n == 0.0 && highest == 0.0 && n.is_sign_positive() && highest.is_sign_negative())
        {
            highest = n;
        }
    }
    Ok(Value::Num(highest))
}

fn math_min_generic(i: &mut Interp, a: &[Value]) -> Result<Value, Value> {
    let mut nums = Vec::with_capacity(a.len());
    for v in a {
        nums.push(ab(i.to_number(v))?);
    }
    let mut lowest = f64::INFINITY;
    for n in nums {
        if n.is_nan() {
            return Ok(Value::Num(f64::NAN));
        }
        if n < lowest
            || (n == 0.0 && lowest == 0.0 && n.is_sign_negative() && lowest.is_sign_positive())
        {
            lowest = n;
        }
    }
    Ok(Value::Num(lowest))
}

/// Correctly-rounded sum of finite f64s, via Shewchuk's nonoverlapping-partials algorithm with
/// CPython's final round-half-to-even step (the `math.fsum` algorithm).
fn fsum_exact(values: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for &xi in values {
        let mut x = xi;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    let n = partials.len();
    if n == 0 {
        return 0.0;
    }
    let mut hi = partials[n - 1];
    let mut lo = 0.0;
    let mut idx = n - 1;
    while idx > 0 {
        idx -= 1;
        let x = hi;
        let y = partials[idx];
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    // Round half to even: nudge when the residual and the next partial agree in sign.
    if idx > 0 && ((lo < 0.0 && partials[idx - 1] < 0.0) || (lo > 0.0 && partials[idx - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}

thread_local! {
    /// Natives whose complete algorithm runs no author code when every argument is a Number
    /// (Math's numeric functions: ToNumber is then the identity). See
    /// `bytecode::INTRINSIC_NUMERIC_ONLY`.
    static NUMERIC_ONLY_NATIVES: std::cell::RefCell<crate::fasthash::FastSet<usize>> =
        std::cell::RefCell::new(Default::default());
}

fn register_numeric_only_native(f: NativeFn) {
    NUMERIC_ONLY_NATIVES.with(|set| set.borrow_mut().insert(f as *const () as usize));
}

thread_local! {
    /// Math functions with an inline call-template implementation, keyed by native identity
    /// (see `bytecode::INTRINSIC_MATH_FIRST`).
    static MATH_INTRINSICS: std::cell::RefCell<crate::fasthash::FastMap<usize, u8>> =
        std::cell::RefCell::new(Default::default());
}

fn register_math_intrinsic(f: NativeFn, id: u8) {
    MATH_INTRINSICS.with(|map| map.borrow_mut().insert(f as *const () as usize, id));
}

/// The inline Math intrinsic id of native `f`, or 0.
pub(crate) fn math_intrinsic(f: NativeFn) -> u8 {
    MATH_INTRINSICS.with(|map| {
        map.borrow()
            .get(&(f as *const () as usize))
            .copied()
            .unwrap_or(0)
    })
}

/// Whether `f` was registered by [`register_numeric_only_native`].
pub(crate) fn is_numeric_only_native(f: NativeFn) -> bool {
    NUMERIC_ONLY_NATIVES.with(|set| set.borrow().contains(&(f as *const () as usize)))
}
