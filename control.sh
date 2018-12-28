#!/bin/bash

if [[ ! -z "$DEMO" ]]; then
	echo "Running on demo environment"
fi

if [[ -z "$ALGORITHMS" ]]; then
	echo "ALGORITHM: Default"
	ALGORITHMS=(nature bootstrap actioncoverage_bellman actioncoverage_swarm actioncoverage_doubleq agentcoverage)
fi

if [[ -z "$RUN_SCRIPT" ]]; then
	RUN_SCRIPT=/home/rchouras/workplace/rishav-internship/code/Atari-experiments/run.sh
fi

if [[ -z "$EXPERIMENTS" ]]; then
	EXPERIMENTS=/home/rchouras/workplace/rishav-internship/code/Atari-experiments/experiments
fi
if [[ -z "$GAMES" ]]; then
	echo "GAMES: Default"
	GAMES=(alien amidar assault asterix asteroids atlantis bank_heist battle_zone beam_rider bowling boxing breakout centipede chopper_command crazy_climber demon_attack double_dunk enduro fishing_derby freeway frostbite gopher gravitar hero ice_hockey jamesbond kangaroo krull kung_fu_master montezuma_revenge ms_pacman name_this_game pong private_eye qbert riverraid road_runner robotank seaquest space_invaders star_gunner tennis time_pilot tutankham up_n_down venture video_pinball wizard_of_wor zaxxon)
fi

if [[ -z "$CUDA_DEVICES" ]]; then
	# Setting the list of cuda_devices
	if [ $(hostname) = 'dgx1' ]; then
		export CUDA_DEVICES=($(seq 0 7))
	elif [[ $(hostname) =~ ^maxwell[0-9]{2}$ ]]; then
		export CUDA_DEVICES=($(seq 0 1))
	else
		export CUDA_DEVICES=($(seq 0 0))
	fi
fi

#################################################################
echo "Would be testing on following algorithms: ${ALGORITHMS[@]}"
echo "Path for the run script: $RUN_SCRIPT"
echo "Save destination path: $EXPERIMENTS"
echo "Following games will be tested: ${GAMES[@]}"
echo "DEVICES ARE: ${CUDA_DEVICES[@]}"

if [[ -z "$DEMO" ]]; then
	gpu_idx=6
	for ALGORITHM in ${ALGORITHMS[@]}; do
		echo "Running algorithm: $ALGORITHM"
		for GAME in ${GAMES[@]}; do
			echo "Init on game: $GAME"
			echo "Command: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$gpu_idx]} nohup bash $RUN_SCRIPT $ALGORITHM $GAME -experiments $EXPERIMENTS/$ALGORITHM $@ > /dev/null 2>&1 &"
			export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$gpu_idx]}
			nohup bash $RUN_SCRIPT $ALGORITHM $GAME -experiments $EXPERIMENTS/$ALGORITHM $@ > /dev/null 2>&1 &
			gpu_idx=$(( $gpu_idx + 1 ))
			[ $gpu_idx -ge ${#CUDA_DEVICES[@]} ] && gpu_idx=0
		done
	done
else
	gpu_idx=0
	for ALGORITHM in ${ALGORITHMS[@]}; do
		echo "Running algorithm: $ALGORITHM"
		echo "Init on game: rlenvs.Catch"
		echo "Command: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$gpu_idx]} nohup bash $RUN_SCRIPT $ALGORITHM demo -experiments $EXPERIMENTS/$ALGORITHM $@ > /dev/null 2>&1 &"
		export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$gpu_idx]}
#		nohup bash $RUN_SCRIPT $ALGORITHM demo -experiments $EXPERIMENTS/$ALGORITHM $@ > /dev/null 2>&1 &
		gpu_idx=$(( $gpu_idx + 1 ))
		[ $gpu_idx -ge ${#CUDA_DEVICES[@]} ] && gpu_idx=0
	done
fi
