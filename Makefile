simu_src = ./simulate/simulate.py ./simulate/battery.py

.PHONY: help run-null run-1 run-2 run-3 run-our

all: help

help:
	@echo 'run-null - run solution with null battery controler (used to test the simulator)'
	@echo 'run-1 - run with first place battery controller'
	@echo 'run-2 - run with second place battery controller'
	@echo 'run-3 - run with third place battery controller'
	@echo 'run-our - run with our battery controller'

run-all: run-1 run-2 run-3 run-our

################
# null controller
run-null: ./all_results/null_results.csv

./all_results/null_results.csv: entrypoint.sh ${simu_src} ./simulate/battery_controller_null.py
	./entrypoint.sh ./simulate/battery_controller_null.py "null"

################
# 1st controller
run-1: ./all_results/1_results.csv

./all_results/1_results.csv: entrypoint.sh ${simu_src} ./1st-place/battery_controller.py
	./entrypoint.sh ./1st-place/battery_controller.py "1"

################
# 2nd controller
run-2: ./all_results/2_results.csv

./all_results/2_results.csv: entrypoint.sh ${simu_src} ./2nd-place/battery_controller.py
	./entrypoint.sh ./2nd-place/battery_controller.py "2" ./2nd-place/assets/

################
# 3rd controller
run-3: ./all_results/3_results.csv

./all_results/3_results.csv: entrypoint.sh ${simu_src} ./3rd-place/battery_controller.py
	./entrypoint.sh ./3rd-place/battery_controller.py "3"

################
# our controller
run-our: ./all_results/our_results.csv

./all_results/our_results.csv: entrypoint.sh ${simu_src} ./our-solution/battery_controller.py
	./entrypoint.sh ./our-solution/battery_controller.py "our" ./our-solution/assets/
