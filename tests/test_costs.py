from bodhan_core.costs import ocr_cost, speech_cost, transcription_cost, translation_cost


def test_cost_formulas_match_assignment_rates():
    assert transcription_cost(60) == 0.10
    assert translation_cost(10_000) == 0.20
    assert speech_cost(10_000) == 6.00
    assert ocr_cost(3) == 0.60


def test_negative_usage_is_clamped_to_zero():
    assert transcription_cost(-1) == 0
    assert translation_cost(-1) == 0
    assert speech_cost(-1) == 0
    assert ocr_cost(-1) == 0
