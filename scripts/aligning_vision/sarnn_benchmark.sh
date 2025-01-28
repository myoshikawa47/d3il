MUJOCO_GL=egl python run_vision.py --config-name=aligning_vision_config \
              --multirun seed=0,1 \
              agents=sarnn_agent \
              agent_name=sarnn_agent \
              window_size=1 \
              +trainset.rnn=True \
              +valset.rnn=True