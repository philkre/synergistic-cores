from syncore.math_eval import evaluate, is_correct, last_boxed, load_subset, normalize


def test_last_boxed_nested_and_last():
    assert last_boxed(r"so \boxed{1} then \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{\left( 3, \frac{\pi}{2} \right)}") == r"\left( 3, \frac{\pi}{2} \right)"
    assert last_boxed("no answer here") is None
    assert last_boxed(r"\boxed{unclosed") is None


def test_normalize_equivalences():
    assert normalize(r"\left( 3, \frac{\pi}{2} \right)") == normalize(r"(3,\frac{\pi}{2})")
    assert normalize(r"\dfrac{1}{2}") == normalize(r"\frac{1}{2}")
    assert normalize(r"10\%") == normalize("10")
    assert normalize(r"5^\circ") == normalize("5")
    assert normalize(r"\text{(C)}") == normalize("(C)")
    assert normalize("7.") == "7"


def test_is_correct():
    assert is_correct(r"The answer is \boxed{\dfrac{1}{2}}.", r"\frac{1}{2}")
    assert not is_correct(r"\boxed{3}", "4")
    assert not is_correct("no box", "4")


def test_load_subset_stratified():
    sub = load_subset(per_level=2, seed=0)
    assert len(sub) == 10
    assert sorted(p["level"] for p in sub) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert load_subset(per_level=2, seed=0) == sub


def test_evaluate_runs_on_tiny(tiny):
    model, tok = tiny
    probs = [{"problem": "What is 1+1?", "answer": "2", "level": 1}]
    res = evaluate(model, tok, probs, max_new_tokens=5, batch_size=1)
    assert set(res) == {"accuracy", "correct", "outputs"} and 0.0 <= res["accuracy"] <= 1.0


def test_load_subset_exclude_makes_disjoint():
    calib = load_subset(per_level=10, seed=1)
    ev = load_subset(per_level=30, seed=0, exclude=calib)
    assert len(ev) == 150
    assert not {p["problem"] for p in calib} & {p["problem"] for p in ev}
