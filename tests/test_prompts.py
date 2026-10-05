from syncore.prompts import PROMPTS, all_prompts


def test_six_categories_of_ten():
    assert list(PROMPTS) == ["syntax", "pos", "numerical", "commonsense", "creative", "social"]
    assert all(len(v) == 10 for v in PROMPTS.values())


def test_all_prompts_order_and_content():
    ps = all_prompts()
    assert len(ps) == 60
    assert ps[0] == ("syntax", "Correct the error: He go to school every day.")
    assert ps[-1] == ("social", "Imagine a scenario where a character has to forgive someone who wronged them.")
