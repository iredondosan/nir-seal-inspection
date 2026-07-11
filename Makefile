# Reproduce the thesis results — one target per result.
# Requires the dataset (data/) and trained weights (models/) locally; see REPRODUCE.md.
# Override the interpreter with:  make PYTHON=.venv/bin/python table-4.1
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install table-4.1 table-4.2 table-4.3 table-4.4 table-4.5 table-4.6 \
        table-4.7 table-4.8 kfold patchcore errors deploy-onnx demo

help:  ## List targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

install:      ## Install the package (editable) + dependencies
	pip install -e .

table-4.1:    ## Seal Dice per product
	$(PYTHON) evaluation/eval_seal.py
table-4.2:    ## Resolution boundary metrics (1280 row)
	$(PYTHON) experiments/ablation_resolution_1280.py
table-4.3:    ## LOPO seal, zero-shot (retrains 5x, GPU, ~85 min)
	$(PYTHON) experiments/lopo_seal.py
table-4.4:    ## Defect threshold sweep (operating point)
	$(PYTHON) evaluation/eval_thresholds.py
table-4.5:    ## Transfer-learning ablation (isolated + end-to-end)
	$(PYTHON) experiments/ablation_transfer.py
	$(PYTHON) experiments/ablation_transfer_e2e.py
table-4.6:    ## TinyUNet capacity ablation
	$(PYTHON) evaluation/eval_tiny_e2e.py
table-4.7:    ## Augmentation ablation (roll / sealjit)
	$(PYTHON) experiments/ablation_augment.py
table-4.8:    ## Deployment CPU latencies
	$(PYTHON) deploy/bench_cpu.py
kfold:        ## 5-fold CV of the defect model (retrains, GPU)
	$(PYTHON) experiments/kfold_cv.py
patchcore:    ## PatchCore unsupervised baseline
	$(PYTHON) experiments/baseline_patchcore.py
errors:       ## Error analysis of hold-out FN/FP (§5.4)
	$(PYTHON) evaluation/error_analysis.py
deploy-onnx:  ## Export ONNX + INT8 quantize + CPU benchmark
	$(PYTHON) deploy/quantize_int8.py
	$(PYTHON) deploy/bench_cpu.py
demo:         ## Launch the interactive Streamlit demo (ONNX, CPU)
	cd demo && streamlit run app.py
