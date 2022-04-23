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

python test.py --cfg neuralvision/configs/task22/baseline_aug1234.py
python -m scripts.run_analysis neuralvision/configs/task22/baseline_aug1234.py
python -m scripts.run_benchmark neuralvision/configs/task22/baseline_aug1234.py