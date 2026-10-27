"""ms-swift plugin (``--external_plugins``) that registers TabPilot's training components."""

from tabpilot.training import ps_opsd, rollout, sa_grpo

rollout.register_template()
rollout.install_rollout_loop()
sa_grpo.register()
ps_opsd.register()
ps_opsd.install()
