from src.providers.paddle_provider import OCRFactory, PaddleOCREngine


def test_paddle_engine_uses_paddleocr_3x_constructor_args(mocker):
    """PaddleOCR 3.x rewrote its constructor: use_angle_cls -> use_textline_orientation,
    show_log removed entirely, use_gpu removed in favor of device="cpu"/"gpu". A prior
    regression passed the old 2.x-era kwargs and broke engine construction outright
    (ValueError: Unknown argument: show_log) — assert the exact kwargs passed to confirm
    this can't silently regress again."""
    mock_paddle_cls = mocker.patch("src.providers.paddle_provider._PaddleOCR")

    PaddleOCREngine(lang="en", use_angle_cls=True, enable_mkldnn=False)

    mock_paddle_cls.assert_called_once_with(
        use_textline_orientation=True,
        lang="en",
        enable_mkldnn=False,
        device="cpu",
    )
    call_kwargs = mock_paddle_cls.call_args.kwargs
    assert "use_gpu" not in call_kwargs
    assert "show_log" not in call_kwargs
    assert "use_angle_cls" not in call_kwargs


def test_factory_paddle_engine_construction_does_not_raise(mocker):
    """End-to-end through the real factory (not just the class directly), forcing a fresh
    instance so the cache from another test can't hide a construction-time regression."""
    mocker.patch("src.providers.paddle_provider._PaddleOCR")
    OCRFactory._instances.pop("paddle", None)

    engine = OCRFactory.get_engine("paddle")

    assert isinstance(engine, PaddleOCREngine)
    OCRFactory._instances.pop("paddle", None)
