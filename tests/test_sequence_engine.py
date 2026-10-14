from vision.sequence_engine import SequenceDef, SequenceEngine, SequenceTracker


def mk(steps=("fist", "peace", "open_palm"), **kw):
    return SequenceTracker(SequenceDef("t", list(steps), "lightning", hold_time=0.3, timeout=2.0, cooldown=3.0, **kw))


def hold(t, pose, start, dur=0.4, step=0.05):
    ev, now = None, start
    while now <= start + dur:
        ev = t.update(pose, now) or ev
        now += step
    return ev, now


def test_progression_and_completion():
    t = mk()
    assert hold(t, "fist", 0)[0] == "advanced" and t.idx == 1
    assert hold(t, "peace", 0.5)[0] == "advanced" and t.idx == 2
    assert hold(t, "open_palm", 1.0)[0] == "completed" and t.idx == 0


def test_hold_time_required():
    t = mk()
    t.update("fist", 0.0)
    assert t.update("fist", 0.1) is None and t.idx == 0


def test_timeout_reset():
    t = mk()
    hold(t, "fist", 0)
    assert t.idx == 1
    assert t.update("none", 5.0) == "reset_timeout" and t.idx == 0


def test_wrong_pose_resets_but_none_is_tolerated():
    t = mk()
    hold(t, "fist", 0)
    assert t.update("none", 0.6) is None and t.idx == 1          # hand briefly lost
    assert t.update("open_palm", 0.7) == "reset_wrong" and t.idx == 0


def test_holding_completed_pose_does_not_reset():
    t = mk()
    hold(t, "fist", 0, dur=1.0)  # keeps holding fist after the step completed
    assert t.idx == 1


def test_same_pose_twice_requires_release():
    t = mk(steps=("fist", "fist"))
    hold(t, "fist", 0, dur=1.0)
    assert t.idx == 1
    t.update("open_palm", 1.2)   # release (non-matching pose after progress would reset...)
    # open_palm is a wrong pose -> reset; use 'none' release instead
    t2 = mk(steps=("fist", "fist"))
    hold(t2, "fist", 0, dur=1.0)
    t2.update("none", 1.1)
    assert hold(t2, "fist", 1.2)[0] == "completed"


def test_cooldown_blocks_repeat():
    t = mk(steps=("fist",))
    assert hold(t, "fist", 0)[0] == "completed"
    t.update("none", 0.6)
    assert t.update("fist", 0.7) is None and t.update("fist", 1.2) is None   # inside 3 s cooldown
    assert hold(t, "fist", 4.0)[0] == "completed"


def test_engine_progress_reports_best():
    e = SequenceEngine([SequenceDef("a", ["fist", "peace"]), SequenceDef("b", ["peace", "fist"])])
    now = 0.0
    while now < 0.5:
        e.update("fist", now)
        now += 0.05
    p = e.progress(now)
    assert p["name"] == "a" and p["done"] == 1 and p["total"] == 2
