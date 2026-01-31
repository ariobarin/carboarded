@echo off
REM Windows batch script to launch multiple RL training experiments
REM Usage: run_experiments.bat

cd racing_sim

echo === Launching SAC Experiment 1.1: Progress 0.75 ===
start "SAC Progress 0.75" cmd /c "py scripts/train.py --algo sac --preset fast --total-timesteps 60000 --config configs/opt_progress_0p75.yaml --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 --random-start --n-envs 4 --vec-env subproc --no-progress --load-model ^"../Good Models/SAC Wavy V2 Random Start 183.7 at 90k/best_model.zip^" 2>&1 | tee logs/exp1p1_sac_progress_0p75_v2.log"

timeout /t 5

echo === Launching PPO Experiment 1.5: Extended 120k ===
start "PPO Extended 120k" cmd /c "py scripts/train.py --algo ppo --preset fast --total-timesteps 120000 --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.04 --n-envs 4 --vec-env subproc --no-progress 2>&1 | tee logs/exp1p5_ppo_extended_120k_v2.log"

echo === Experiments launched in separate windows ===
echo Check logs in racing_sim/logs/
pause
