//! Exhaustion must disable layout proofs, not change OrdinaryGet/OrdinarySet.
//! ECMA-262 snapshot e28783d5fc9d, #sec-ordinaryget / #sec-ordinarysetwithowndescriptor.
use super::*;
use crate::value::SHAPE_UNCACHEABLE as UNKNOWN;
use std::cell::{Cell, RefCell};

fn state() -> IcState {
    IcState {
        recv_shape: 17,
        holder_shape: 23,
        depth: 0,
        ..IcState::EMPTY
    }
}

#[test]
fn uncacheable_shapes_never_publish_own_array_or_intermediate_guards() {
    assert!(!IcState::EMPTY.has_cacheable_shapes());
    assert!(
        IcState {
            recv_shape: UNKNOWN - 1,
            holder_shape: UNKNOWN - 1,
            ..state()
        }
        .has_cacheable_shapes(),
        "the final real identity stays usable"
    );
    for depth in 0..=IC_MAX_DEPTH {
        for array in [0, IC_ARR_KEYCHK, IC_ACCESSOR, IC_ARR_KEYCHK | IC_ACCESSOR] {
            let valid = IcState {
                depth: depth | array,
                mid_ok: 15,
                mid_shape: 2,
                mid2_shape: 3,
                mid3_shape: 4,
                mid4_shape: 5,
                ..state()
            };
            assert!(valid.has_cacheable_shapes());
            for word in 0..6 {
                let mut invalid = valid;
                match word {
                    0 => invalid.recv_shape = UNKNOWN,
                    1 => invalid.holder_shape = UNKNOWN,
                    2 => invalid.mid_shape = UNKNOWN,
                    3 => invalid.mid2_shape = UNKNOWN,
                    4 => invalid.mid3_shape = UNKNOWN,
                    _ => invalid.mid4_shape = UNKNOWN,
                }
                assert!(
                    !invalid.has_cacheable_shapes(),
                    "depth {depth}, word {word}"
                );
            }
        }
    }
    // Unrecorded words are not proof-bearing. Empty shape zero remains cacheable.
    assert!(IcState {
        recv_shape: 0,
        holder_shape: 0,
        mid_shape: UNKNOWN,
        mid2_shape: UNKNOWN,
        mid3_shape: UNKNOWN,
        mid4_shape: UNKNOWN,
        ..state()
    }
    .has_cacheable_shapes());
}

#[test]
fn uncacheable_shapes_distinguish_absence_levels_from_creation_metadata() {
    for levels in 1..=6 {
        for bad in 0..6 {
            let mut shapes = [17; 6];
            shapes[bad] = UNKNOWN;
            let absent = IcState {
                depth: IC_ABSENT,
                slot: levels,
                recv_shape: shapes[0],
                mid_shape: shapes[1],
                mid2_shape: shapes[2],
                mid3_shape: shapes[3],
                mid4_shape: shapes[4],
                holder_shape: shapes[5],
                mid_ok: 0,
            };
            assert_eq!(absent.has_cacheable_shapes(), bad >= levels as usize);
        }
    }
    for levels in [0, 7, u32::MAX] {
        assert!(!IcState {
            depth: IC_ABSENT,
            slot: levels,
            ..state()
        }
        .has_cacheable_shapes());
    }
    let creation = IcState {
        depth: IC_CREATE,
        mid_shape: UNKNOWN,
        slot: UNKNOWN,
        mid2_shape: UNKNOWN,
        mid3_shape: UNKNOWN,
        mid4_shape: UNKNOWN,
        ..state()
    };
    assert!(
        creation.has_cacheable_shapes(),
        "epoch and pointer words are not shapes"
    );
    assert!(!IcState {
        recv_shape: UNKNOWN,
        ..creation
    }
    .has_cacheable_shapes());
    assert!(!IcState {
        holder_shape: UNKNOWN,
        ..creation
    }
    .has_cacheable_shapes());
}

#[test]
fn uncacheable_shapes_do_not_allocate_computed_cache_or_replace_valid_ways() {
    let mut cache = ComputedReadCache::default();
    let key = crate::lstr::LStr::from("field");
    let invalid = IcState {
        recv_shape: UNKNOWN,
        ..state()
    };
    cache.insert(&key, invalid);
    assert!(cache.raw_sets.is_null());
    assert_eq!(key.strong_count(), 1);
    cache.insert(&key, state());
    cache.insert(&key, invalid);
    assert!(cache.lookup(key.as_ptr() as usize, UNKNOWN).is_none());
    assert_eq!(
        cache
            .lookup(key.as_ptr() as usize, 17)
            .unwrap()
            .holder_shape,
        23
    );
    assert_eq!(key.strong_count(), 2);
}

#[test]
fn uncacheable_shapes_are_generic_in_ic_and_runtime_feedback() {
    use crate::feedback::{
        CurrentPropertyTrace, FeedbackVector, ObservationKind as Kind, ObservationRole as Role,
        ObservationState as State, PropertyOutcome,
    };
    for raw_trace in [false, true] {
        for (receiver, holder) in [(UNKNOWN, 23), (17, UNKNOWN)] {
            let (layout, bindings) = feedback_layout_for_ops(&[Op::GetProp(0, 0)], &[]);
            let feedback = FeedbackVector::new_with_enabled(layout, bindings, true);
            let shapes = RefCell::new(Vec::new());
            if raw_trace {
                feedback.observe_current_property(
                    0,
                    &shapes,
                    CurrentPropertyTrace {
                        receiver_shape: Some(receiver),
                        holder_shape: Some(holder),
                        outcome: Some(PropertyOutcome::Data),
                        field_slot: Some(0),
                        ..CurrentPropertyTrace::default()
                    },
                );
            } else {
                let caches = [Cell::new(IcState {
                    recv_shape: receiver,
                    holder_shape: holder,
                    ..state()
                })];
                refresh_current_layout_feedback(&feedback, &shapes, &caches);
            }
            let site = feedback.sites().next().unwrap().0;
            for (kind, role) in [
                (Kind::ReceiverLayout, Role::Receiver),
                (Kind::HolderLayout, Role::Holder),
                (Kind::PropertyAccess, Role::Access),
            ] {
                assert_eq!(feedback.read(site, kind, role).state(), State::Generic);
            }
            assert!(
                shapes.borrow().is_empty(),
                "the sentinel must never become a layout token"
            );
        }
    }
}

#[test]
fn uncacheable_shapes_cannot_seed_initializer_transition_plans() {
    let plan = InitializerPlan {
        fields: vec![InitializerField {
            slot: 0,
            default: None,
            name: 0,
            cache: 0,
        }],
        shapes: Cell::default(),
    };
    plan.cache_shapes(7, 64, 17, &[23]);
    assert_eq!(plan.cached_shapes(7, 64, 17).unwrap()[0], 23);
    plan.cache_shapes(7, 64, UNKNOWN, &[23]);
    assert!(plan.cached_shapes(7, 64, UNKNOWN).is_none());
    plan.cache_shapes(7, 64, 17, &[UNKNOWN]);
    assert_eq!(plan.cached_shapes(7, 64, 17).unwrap()[0], 23);
}
