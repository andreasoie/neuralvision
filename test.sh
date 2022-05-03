# python -m scripts.run_analysis neuralvision/configs/retina_P1.py
# python -m scripts.run_benchmark neuralvision/configs/retina_P1.py

# python -m scripts.run_analysis neuralvision/configs/retina_P2.py
# python -m scripts.run_benchmark neuralvision/configs/retina_P2.py

# python -m scripts.run_analysis neuralvision/configs/retina_P3.py
# python -m scripts.run_benchmark neuralvision/configs/retina_P3.py

# python -m scripts.run_analysis neuralvision/configs/retina_P4.py
# python -m scripts.run_benchmark neuralvision/configs/retina_P4.py

# python test.py --cfg neuralvision/configs/retina_P1.py
# python test.py --cfg neuralvision/configs/retina_P2.py
# python test.py --cfg neuralvision/configs/retina_P3.py
# python test.py --cfg neuralvision/configs/retina_P4.py

# python test.py --cfg neuralvision/configs/template/tdt4265.py

# python -m scripts.run_analysis core/configs/task23/retina_P4.py
# python -m scripts.run_benchmark core/configs/task23/retina_P4.py

# python -m scripts.run_analysis core/configs/task25/retina_P4_retrain.py
# python -m scripts.run_benchmark core/configs/task25/retina_P4_retrain.py

# python -m scripts.run_analysis core/configs/task25/retina_P4_retrain2.py
# python -m scripts.run_benchmark core/configs/task25/retina_P4_retrain2.py

python -m scripts.save_comparison_images core/configs/task25/retina_P4_retrain.py -n 6 -c 0.8
python -m scripts.save_comparison_images core/configs/task25/retina_P4_retrain2.py -n 6 -c 0.8

