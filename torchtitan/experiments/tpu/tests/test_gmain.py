from unittest import mock

from torchtitan.experiments.tpu import gmain


def test_handle_main_uses_tpu_argument_parser():
    start_trainer = mock.Mock()
    with mock.patch.object(gmain.app, "run") as app_run:
        gmain.handle_main(start_trainer)

    assert gmain.global_start_trainer_func is start_trainer
    app_run.assert_called_once_with(gmain.main, flags_parser=gmain._parse_flags)
