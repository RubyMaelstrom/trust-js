//! ECMA-262 OrdinaryGet/OrdinarySetWithOwnDescriptor, Array [[DefineOwnProperty]],
//! Number multiplication and ToInt32. Official local snapshot e28783d5 (2026-09-06).
//! A physical numeric view must preserve descriptors/owners and invalidate loop facts on aliases.

use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn array(engine: &mut Engine, name: &str) -> crate::value::Gc {
    let global = crate::value::Value::Obj(engine.interp.global.clone());
    engine
        .interp
        .get_member_key(&global, name)
        .unwrap_or_else(|_| panic!("{name}"))
        .as_obj()
        .unwrap()
        .clone()
}

fn reset_entries() {
    #[cfg(target_arch = "aarch64")]
    super::TEST_NUMERIC_REGION_ENTRIES.with(|count| count.set(0));
}

fn entered(tier: Tier) {
    #[cfg(target_arch = "aarch64")]
    if tier == Tier::Jit {
        assert!(
            super::TEST_NUMERIC_REGION_ENTRIES.with(|count| count.get()) > 0,
            "receiver/numeric guards must admit the actual region"
        );
    }
    let _ = tier;
}

#[test]
fn numeric_array_shift_keeps_warm_native_views_coherent_across_gc_and_append() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for literal in ["[1,2,3]", "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]"] {
            let len = if literal == "[1,2,3]" { 3 } else { 16 };
            let total = len * (len + 1) / 2;
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(&mut engine, &format!(
                "var a={literal};a.named={{id:99}};function shiftScan(a,n){{var sum=0;for(var k=0;k<n;k++)sum+=a[k];return sum;}}"
            ));
            reset_entries();
            assert_eq!(
                eval(&mut engine, &format!("shiftScan(a,{len})")),
                total.to_string()
            );
            entered(tier);
            assert_eq!(eval(&mut engine, "a.shift()"), "1");
            engine
                .interp
                .gc_collect_young(crate::value::GcCause::Explicit);
            engine.interp.gc_collect();
            reset_entries();
            assert_eq!(
                eval(&mut engine, &format!("shiftScan(a,{})", len - 1)),
                (total - 1).to_string()
            );
            entered(tier);
            assert_eq!(eval(&mut engine, "a.push(99);a.named.id"), "99");
            reset_entries();
            assert_eq!(
                eval(&mut engine, &format!("shiftScan(a,{len})")),
                (total + 98).to_string()
            );
            entered(tier);
            assert_eq!(eval(&mut engine, "a.shift()"), "2");
            engine.interp.gc_collect();
            reset_entries();
            assert_eq!(
                eval(&mut engine, &format!("shiftScan(a,{})", len - 1)),
                (total + 96).to_string()
            );
            entered(tier);
        }
    }
}

#[test]
fn numeric_front_edits_reload_native_views_within_the_same_activation() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            var frontValues=[];
            for(var k=1;k<=64;k++)frontValues.push(k);
            function editFront(a,n){
                var total=0;
                for(var k=0;k<n;k++){
                    total+=a[0];
                    var first=a.shift();
                    total+=a[0];
                    a.unshift(first);
                    total+=a[0];
                    a.push(first+20);
                    total+=a.pop();
                }
                return total;
            }
        "#,
        );
        for _ in 0..8 {
            reset_entries();
            assert_eq!(eval(&mut engine, "editFront(frontValues,80)"), "2000");
            assert_eq!(
                eval(
                    &mut engine,
                    "frontValues.length+':'+frontValues[0]+':'+frontValues[63]"
                ),
                "64:1:64"
            );
            engine
                .interp
                .gc_collect_young(crate::value::GcCause::Explicit);
            engine.interp.gc_collect();
        }
    }
}

#[test]
fn numeric_array_length_coercion_gc_revalidates_warm_views_after_shrink_and_growth() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for assignment in [true, false] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            engine.interp.def_method(
                &engine.interp.global,
                "collectArrayLengthTest",
                0,
                |interp, _, _| {
                    interp.gc_collect();
                    Ok(crate::value::Value::Undefined)
                },
            );
            eval(
                &mut engine,
                r#"
                var a=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
                a.named={value:99};
                function scanLengthArray(a,n){var sum=0;for(var k=0;k<n;k++)sum+=a[k];return sum;}
                function readLengthArray(a){return a.length;}
                var coercions=0;
                var lengthValue={valueOf(){
                    collectArrayLengthTest();coercions++;
                    if(a.named.value!==99)throw 'lost named owner';
                    return 8;
                }};
                "#,
            );
            reset_entries();
            assert_eq!(eval(&mut engine, "scanLengthArray(a,16)"), "136");
            entered(tier);
            assert_eq!(eval(&mut engine, "readLengthArray(a)"), "16");
            eval(
                &mut engine,
                if assignment {
                    "a.length=lengthValue"
                } else {
                    "Object.defineProperty(a,'length',{value:lengthValue})"
                },
            );
            assert_eq!(eval(&mut engine, "coercions+':'+readLengthArray(a)"), "2:8");
            reset_entries();
            assert_eq!(eval(&mut engine, "scanLengthArray(a,8)"), "36");
            entered(tier);
            eval(&mut engine, "a.length=16;collectArrayLengthTest();");
            assert_eq!(eval(&mut engine, "readLengthArray(a)"), "16");
            assert_eq!(
                eval(&mut engine, "Number.isNaN(scanLengthArray(a,16))"),
                "true"
            );
            // Replacing the holes rebuilds a usable mirror. Both the old machine entry and
            // named length cache must observe current storage, not the pre-shrink allocation.
            eval(
                &mut engine,
                "for(var j=8;j<16;j++)a[j]=j+1;collectArrayLengthTest();",
            );
            reset_entries();
            assert_eq!(eval(&mut engine, "scanLengthArray(a,16)"), "136");
            entered(tier);
            assert_eq!(eval(&mut engine, "a.named.value"), "99");
        }
    }
}

#[test]
fn numeric_array_views_admit_inline_and_heap_packed_without_moving_named_slots() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for size in [3, 16] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            let elements = (0..size)
                .map(|index| format!("{}.25", index))
                .collect::<Vec<_>>()
                .join(",");
            eval(
                &mut engine,
                &format!(
                    r#"
                function make(){{return [{elements}];}}
                var numeric=make(),symbol=Symbol('key');numeric.named='kept';numeric[symbol]=numeric;
                function scan(a,n){{let sum=0;for(let k=0;k<n;k++)sum+=a[k];return sum;}}
            "#
                ),
            );
            let object = array(&mut engine, "numeric");
            let (length_slot, named_slot, shape) = {
                let body = object.borrow();
                (
                    body.props.slot_of("length"),
                    body.props.slot_of("named"),
                    body.props.shape(),
                )
            };
            reset_entries();
            let expected = size * (size - 1) / 2;
            assert_eq!(
                eval(&mut engine, &format!("scan(numeric,{size})")),
                (expected as f64 + size as f64 * 0.25).to_string()
            );
            entered(tier);
            let body = object.borrow();
            assert_eq!(
                (
                    body.props.slot_of("length"),
                    body.props.slot_of("named"),
                    body.props.shape()
                ),
                (length_slot, named_slot, shape)
            );
            #[cfg(target_arch = "aarch64")]
            if tier == Tier::Jit {
                for index in 0..size {
                    assert_eq!(body.props.mirror_get(index), Some(index as f64 + 0.25));
                }
            }
            drop(body);
            assert_eq!(
                eval(
                    &mut engine,
                    "numeric.named==='kept'&&numeric[symbol]===numeric"
                ),
                "true"
            );
        }
    }
}

#[test]
fn numeric_array_alias_writes_revalidate_integer_facts_before_dependent_arithmetic() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for packed in [false, true] {
            for alias in [false, true] {
                for (value, aliased) in [
                    ("0.5", "16"),
                    ("-0", "0"),
                    ("NaN", "0"),
                    ("Infinity", "0"),
                    ("-Infinity", "0"),
                    ("2147483648", "0"),
                ] {
                    let mut engine = Engine::new();
                    engine.set_tier(tier);
                    engine.set_tier_threshold(0);
                    let make = if packed {
                        "return [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];"
                    } else {
                        "let a=[];for(let k=0;k<16;k++)a[k]=1;return a;"
                    };
                    eval(
                        &mut engine,
                        &format!(
                            r#"
                        function make(){{{make}}}
                        var a=make(),b={};
                        function mixed(a,b,value){{let sum=0;for(let k=0;k<16;k++){{
                            b[k]=value;sum+=(a[k]*2)|0;
                        }}return sum;}}
                    "#,
                            if alias { "a" } else { "make()" }
                        ),
                    );
                    reset_entries();
                    assert_eq!(
                        eval(&mut engine, &format!("mixed(a,b,{value})")),
                        if alias { aliased } else { "32" },
                        "{tier:?} packed={packed} alias={alias} value={value}"
                    );
                    entered(tier);
                }
            }
        }
    }
}

#[test]
fn numeric_array_integer_proofs_exclude_positive_two_to_the_31_boundary() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            var a=[];for(let k=0;k<16;k++)a[k]=1;
            function boundary(a){let sum=0;for(let k=0;k<16;k++){
                a[k]=32768*65536;sum+=(a[k]-2147483648)|0;
            }return sum;}
        "#,
        );
        reset_entries();
        assert_eq!(eval(&mut engine, "boundary(a)"), "0", "{tier:?}");
        entered(tier);
    }
}

#[test]
fn numeric_array_nested_loops_and_checked_gc_exits_preserve_owners_and_writes() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.interp.def_method(
            &engine.interp.global,
            "collectNumericTest",
            0,
            |interp, _, _| {
                interp.gc_collect();
                Ok(crate::value::Value::Undefined)
            },
        );
        eval(
            &mut engine,
            r#"
            function make(){return [1,1,1,1,1,1,1,1];}
            var a=make(),saved=a; a.self=a;
            function nested(a,b){let sum=0;for(let p=0;p<3;p++){
                for(let k=0;k<8;k++){b[k]=0.5;sum+=(a[k]*2)|0;}
            }return sum;}
            function scan(a){let sum=0;for(let k=0;k<8;k++)sum+=a[k];return sum;}
        "#,
        );
        reset_entries();
        assert_eq!(eval(&mut engine, "nested(a,a)"), "24");
        entered(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var before=scan(a),reads=0;
            Object.defineProperty(a,'3',{get(){reads++;collectNumericTest();return 7;},configurable:true});
            a=null;
            var after=scan(saved);
            [before,after,reads,saved.self===saved].join('|');
        "#
            ),
            "4|10.5|1|true",
            "{tier:?}"
        );
        engine.interp.gc_collect();
    }
}

#[test]
fn numeric_array_view_invalidations_preserve_descriptors_holes_and_length() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function make(){return [1,2,3];}
            function scan(a){let n=0;for(let k=0;k<3;k++)n+=a[k];return n;}
            function set(a,k,v){a[k]=v;return a[k];}
            var a=make(),out=[scan(a)];
            out.push(set(a,0,-0),1/a[0]);out.push(scan(a));
            a[1]=NaN;out.push(Number.isNaN(scan(a)));a[1]=2;
            Object.defineProperty(a,'1',{get(){return 10},configurable:true});out.push(scan(a));
            delete a[1];Object.defineProperty(Array.prototype,'1',{get(){return 20},configurable:true});
            var hole=scan(a);delete Array.prototype[1];out.push(hole);
            Object.defineProperty(a,'1',{value:2,writable:true,enumerable:true,configurable:true});
            out.push(scan(a));a.length=1;out.push(a.length,1 in a,2 in a);a.push(4,5);out.push(scan(a));
            Object.seal(a);out.push(scan(a),set(a,1,6),scan(a));
            Object.freeze(a);out.push(scan(a),set(a,1,99),scan(a));
            out.join('|');
        "#
            ),
            "6|0|-Infinity|5|true|13|23|5|1|false|false|9|9|6|11|11|6|11",
            "{tier:?}"
        );
    }
}

#[test]
fn numeric_array_existing_hole_filling_region_invalidates_prepared_views() {
    check_hole_filling_region_storage(true);
}

#[test]
fn numeric_array_existing_hole_filling_region_maintains_classic_views() {
    check_hole_filling_region_storage(false);
}

fn check_hole_filling_region_storage(packed: bool) {
    // Ordinary cold literals now also use contiguous elements. The same test
    // with TRUST_JS_DENSE_ELEMENTS=0 retains explicit coverage of the classic path.
    let packed = packed || crate::value::dense_elements_enabled();
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(if tier == Tier::Jit {
            Tier::Bytecode
        } else {
            tier
        });
        // Compile with warmed property/name feedback, as hot page code does.
        engine.set_tier_threshold(32);
        eval(
            &mut engine,
            &r#"
            var LIMIT=4;
            function Worker(){this.v=0;}
            Worker.prototype.fill=function(packet){var i=0;while(i<LIMIT){
                this.v++;if(this.v>26)this.v=1;packet.a[i]=this.v;i++;
            }};
            function scan(a){let sum=0;for(let k=0;k<4;k++)sum+=a[k];return sum;}
            // Cold Script execution need not choose packed literal storage. Warm the
            // actual factory so this regression explicitly exercises raw packed writes.
            function packedFactory(){return [1,2,3,4];}
            for(var warm=0;warm<40;warm++)packedFactory();
            var worker=new Worker,packet={a:ARRAY_FACTORY},fillArray=packet.a;
            for(var n=0;n<600;n++){worker.fill(packet);scan(packet.a);}
        "#
            .replace(
                "ARRAY_FACTORY",
                if packed {
                    "packedFactory()"
                } else {
                    "[1,2,3,4]"
                },
            ),
        );
        engine.set_tier(tier);
        reset_entries();
        assert_eq!(eval(&mut engine, "scan(packet.a)"), "26");
        entered(tier);
        let object = array(&mut engine, "fillArray");
        #[cfg(target_arch = "aarch64")]
        if tier == Tier::Jit {
            assert_eq!(
                object.borrow().props.packed_values().count(),
                if packed { 4 } else { 0 },
                "fixture must exercise the selected packed or classic storage contract"
            );
            assert_eq!(object.borrow().props.mirror_get(0), Some(5.0));
        }
        assert_eq!(
            eval(&mut engine, "worker.fill(packet);packet.a.join(',')"),
            "9,10,11,12"
        );
        // A write either maintains the numeric mirror or invalidates it; it never leaves it stale.
        assert!(matches!(
            object.borrow().props.mirror_get(0),
            None | Some(9.0)
        ));
        drop(object);
        assert_eq!(eval(&mut engine, "scan(packet.a)"), "42");
        assert_eq!(
            eval(
                &mut engine,
                r#"
            Object.defineProperty(packet.a,'length',{writable:false});
            worker.fill(packet);[scan(packet.a),packet.a.length].join('|');
        "#
            ),
            "58|4"
        );
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn numeric_array_native_numeric_writes_end_failed_preparation_retry_suppression() {
    // OrdinarySetWithOwnDescriptor writes the canonical property as before. The
    // failed-view hint is only a representation policy: every indexed mutation
    // must permit a fresh bounded preparation, including native checked hits.
    for (store, expected) in [
        ("a[k]=v;", "8.5"),
        ("return a[k]=v;", "8.5"),
        ("(k<99?a:[])[k]=v;", "8.5"),
        ("return (k<99?a:[])[k]=v;", "8.5"),
        ("a[k]=v+1;return a[k]+1;", "9.5"),
    ] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            &format!(
                r#"
            function make(){{return [1,2,3];}}
            var numeric=make();
            function store(a,k,v){{{store}}}
            for(var warm=0;warm<100;warm++)store(numeric,0,warm);
        "#
            ),
        );
        let object = array(&mut engine, "numeric");
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        let flags_offset = layout.obj_props + layout.props_mirror_flags;
        {
            let mut body = object.borrow_mut();
            // Model the postcondition of fallible mirror allocation failure without
            // exhausting host memory. Canonical data remain valid numeric Properties.
            let base = (&mut *body as *mut crate::value::Object).cast::<u8>();
            unsafe {
                base.add(flags_offset)
                    .write(crate::value::MIRROR_PACKED_FAILED);
            }
        }
        crate::bytecode::TEST_JIT_SET_ELEM_HELPERS.with(|count| count.set(0));
        crate::bytecode::TEST_JIT_EXEC_ELEMENT_HELPERS.with(|count| count.set(0));
        assert_eq!(
            eval(&mut engine, "store(numeric,0,8.5);numeric[0]"),
            expected
        );
        assert_eq!(
            crate::bytecode::TEST_JIT_SET_ELEM_HELPERS.with(|count| count.get()),
            0,
            "the native Number-to-Number store itself must end retry suppression: {store}"
        );
        assert_eq!(
            crate::bytecode::TEST_JIT_EXEC_ELEMENT_HELPERS.with(|count| count.get()),
            0,
            "a numeric-chain helper bailout must not substitute for native store proof: {store}"
        );
        let mut body = object.borrow_mut();
        let base = (&*body as *const crate::value::Object).cast::<u8>();
        assert_eq!(unsafe { base.add(flags_offset).read() }, 0, "{store}");
        assert!(body.props.prepare_packed_numeric_mirror(), "{store}");
        assert_eq!(body.props.mirror_get(0).unwrap().to_string(), expected);
    }
}
