source env/bin/activate

python branin/objective.py --seed 42
python branin/main.py --seed 42 --savepath ./branin/docs/final/alpha_seed=42.pkl
python branin/main.py --seed 42 --alpha 0.0 --savepath ./branin/docs/final/alpha=0.0_seed=42.pkl
python branin/main.py --seed 42 --alpha 0.2 --savepath ./branin/docs/final/alpha=0.2_seed=42.pkl
python branin/main.py --seed 42 --alpha 0.5 --savepath ./branin/docs/final/alpha=0.5_seed=42.pkl
python branin/main.py --seed 42 --alpha 0.8 --savepath ./branin/docs/final/alpha=0.8_seed=42.pkl
python branin/main.py --seed 42 --alpha 1.0 --savepath ./branin/docs/final/alpha=1.0_seed=42.pkl

python branin/objective.py --seed 43
python branin/main.py --seed 43 --savepath ./branin/docs/final/alpha_seed=43.pkl
python branin/main.py --seed 43 --alpha 0.0 --savepath ./branin/docs/final/alpha=0.0_seed=43.pkl
python branin/main.py --seed 43 --alpha 0.2 --savepath ./branin/docs/final/alpha=0.2_seed=43.pkl
python branin/main.py --seed 43 --alpha 0.5 --savepath ./branin/docs/final/alpha=0.5_seed=43.pkl
python branin/main.py --seed 43 --alpha 0.8 --savepath ./branin/docs/final/alpha=0.8_seed=43.pkl
python branin/main.py --seed 43 --alpha 1.0 --savepath ./branin/docs/final/alpha=1.0_seed=43.pkl

python branin/objective.py --seed 44
python branin/main.py --seed 44 --savepath ./branin/docs/final/alpha_seed=44.pkl
python branin/main.py --seed 44 --alpha 0.0 --savepath ./branin/docs/final/alpha=0.0_seed=44.pkl
python branin/main.py --seed 44 --alpha 0.2 --savepath ./branin/docs/final/alpha=0.2_seed=44.pkl
python branin/main.py --seed 44 --alpha 0.5 --savepath ./branin/docs/final/alpha=0.5_seed=44.pkl
python branin/main.py --seed 44 --alpha 0.8 --savepath ./branin/docs/final/alpha=0.8_seed=44.pkl
python branin/main.py --seed 44 --alpha 1.0 --savepath ./branin/docs/final/alpha=1.0_seed=44.pkl

python branin/objective.py --seed 45
python branin/main.py --seed 45 --savepath ./branin/docs/final/alpha_seed=45.pkl
python branin/main.py --seed 45 --alpha 0.0 --savepath ./branin/docs/final/alpha=0.0_seed=45.pkl
python branin/main.py --seed 45 --alpha 0.2 --savepath ./branin/docs/final/alpha=0.2_seed=45.pkl
python branin/main.py --seed 45 --alpha 0.5 --savepath ./branin/docs/final/alpha=0.5_seed=45.pkl
python branin/main.py --seed 45 --alpha 0.8 --savepath ./branin/docs/final/alpha=0.8_seed=45.pkl
python branin/main.py --seed 45 --alpha 1.0 --savepath ./branin/docs/final/alpha=1.0_seed=45.pkl

python branin/objective.py --seed 46
python branin/main.py --seed 46 --savepath ./branin/docs/final/alpha_seed=46.pkl
python branin/main.py --seed 46 --alpha 0.0 --savepath ./branin/docs/final/alpha=0.0_seed=46.pkl
python branin/main.py --seed 46 --alpha 0.2 --savepath ./branin/docs/final/alpha=0.2_seed=46.pkl
python branin/main.py --seed 46 --alpha 0.5 --savepath ./branin/docs/final/alpha=0.5_seed=46.pkl
python branin/main.py --seed 46 --alpha 0.8 --savepath ./branin/docs/final/alpha=0.8_seed=46.pkl
python branin/main.py --seed 46 --alpha 1.0 --savepath ./branin/docs/final/alpha=1.0_seed=46.pkl

python branin/plot_results.py
