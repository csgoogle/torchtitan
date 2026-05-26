from typing import Any
import os as os
import torchtitan.tools.logging as torchtitan_logging
import torchtitan.tools.profiler as profiler_tool

xprof_session = None

logger = torchtitan_logging.logger


class InternalXProfProfiler:
  """Wrapper for Google-internal XProf session."""

  def __init__(self, profiler_instance: profiler_tool.Profiler):
    self.prof = profiler_instance
    self.session = None

  def start(self) -> None:
    if not xprof_session:
      return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
      logger.info(
          "Starting Internal XProf trace at step %d",
          # pytype: disable=attribute-error
          self.prof.tpu_step_num,
          # pytype: enable=attribute-error
      )
      try:
        self.session = xprof_session.XprofSession()
        self.session.start_session(
            enable_python_tracer=True,
            host_trace_level=3,
            host_cpu_profile=True,
        )
      except RuntimeError as e:
        logger.warning("Failed to start Internal XProf session: %s", e)

  def stop(self) -> None:
    if self.session:
      local_rank = int(os.environ.get("LOCAL_RANK", 0))
      if local_rank == 0:
        logger.info(
            "Stopping Internal XProf profiler at step %d",
            # pytype: disable=attribute-error
            self.prof.tpu_step_num,
            # pytype: enable=attribute-error
        )
        try:
          session_any: Any = self.session
          url = session_any.end_session_and_get_url(
              # pytype: disable=attribute-error
              tag=f"step_{self.prof.tpu_step_num}"
              # pytype: enable=attribute-error
          )
          logger.info(
              "Internal XProf URL for step %d: %s",
              # pytype: disable=attribute-error
              self.prof.tpu_step_num,
              # pytype: enable=attribute-error
              url,
          )
        except RuntimeError as e:
          logger.warning("Failed to stop Internal XProf session: %s", e)
        finally:
          self.session = None
