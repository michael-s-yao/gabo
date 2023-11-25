source env/bin/activate

python mnist/vae.py --seed 42
python mnist/energy.py --seed 42
python mnist/main.py --seed 42 --savepath ./mnist/docs/final/alpha_seed=42.pkl
python mnist/main.py --seed 42 --alpha 0.0 --savepath ./mnist/docs/final/alpha=0.0_seed=42.pkl
python mnist/main.py --seed 42 --alpha 0.2 --savepath ./mnist/docs/final/alpha=0.2_seed=42.pkl
python mnist/main.py --seed 42 --alpha 0.5 --savepath ./mnist/docs/final/alpha=0.5_seed=42.pkl
python mnist/main.py --seed 42 --alpha 0.8 --savepath ./mnist/docs/final/alpha=0.8_seed=42.pkl
python mnist/main.py --seed 42 --alpha 1.0 --savepath ./mnist/docs/final/alpha=1.0_seed=42.pkl

python mnist/vae.py --seed 43
python mnist/energy.py --seed 43
python mnist/main.py --seed 43 --savepath ./mnist/docs/final/alpha_seed=43.pkl
python mnist/main.py --seed 43 --alpha 0.0 --savepath ./mnist/docs/final/alpha=0.0_seed=43.pkl
python mnist/main.py --seed 43 --alpha 0.2 --savepath ./mnist/docs/final/alpha=0.2_seed=43.pkl
python mnist/main.py --seed 43 --alpha 0.5 --savepath ./mnist/docs/final/alpha=0.5_seed=43.pkl
python mnist/main.py --seed 43 --alpha 0.8 --savepath ./mnist/docs/final/alpha=0.8_seed=43.pkl
python mnist/main.py --seed 43 --alpha 1.0 --savepath ./mnist/docs/final/alpha=1.0_seed=43.pkl

python mnist/vae.py --seed 44
python mnist/energy.py --seed 44
python mnist/main.py --seed 44 --savepath ./mnist/docs/final/alpha_seed=44.pkl
python mnist/main.py --seed 44 --alpha 0.0 --savepath ./mnist/docs/final/alpha=0.0_seed=44.pkl
python mnist/main.py --seed 44 --alpha 0.2 --savepath ./mnist/docs/final/alpha=0.2_seed=44.pkl
python mnist/main.py --seed 44 --alpha 0.5 --savepath ./mnist/docs/final/alpha=0.5_seed=44.pkl
python mnist/main.py --seed 44 --alpha 0.8 --savepath ./mnist/docs/final/alpha=0.8_seed=44.pkl
python mnist/main.py --seed 44 --alpha 1.0 --savepath ./mnist/docs/final/alpha=1.0_seed=44.pkl

python mnist/vae.py --seed 45
python mnist/energy.py --seed 45
python mnist/main.py --seed 45 --savepath ./mnist/docs/final/alpha_seed=45.pkl
python mnist/main.py --seed 45 --alpha 0.0 --savepath ./mnist/docs/final/alpha=0.0_seed=45.pkl
python mnist/main.py --seed 45 --alpha 0.2 --savepath ./mnist/docs/final/alpha=0.2_seed=45.pkl
python mnist/main.py --seed 45 --alpha 0.5 --savepath ./mnist/docs/final/alpha=0.5_seed=45.pkl
python mnist/main.py --seed 45 --alpha 0.8 --savepath ./mnist/docs/final/alpha=0.8_seed=45.pkl
python mnist/main.py --seed 45 --alpha 1.0 --savepath ./mnist/docs/final/alpha=1.0_seed=45.pkl

python mnist/vae.py --seed 46
python mnist/energy.py --seed 46
python mnist/main.py --seed 46 --savepath ./mnist/docs/final/alpha_seed=46.pkl
python mnist/main.py --seed 46 --alpha 0.0 --savepath ./mnist/docs/final/alpha=0.0_seed=46.pkl
python mnist/main.py --seed 46 --alpha 0.2 --savepath ./mnist/docs/final/alpha=0.2_seed=46.pkl
python mnist/main.py --seed 46 --alpha 0.5 --savepath ./mnist/docs/final/alpha=0.5_seed=46.pkl
python mnist/main.py --seed 46 --alpha 0.8 --savepath ./mnist/docs/final/alpha=0.8_seed=46.pkl
python mnist/main.py --seed 46 --alpha 1.0 --savepath ./mnist/docs/final/alpha=1.0_seed=46.pkl

python mnist/plot_results.py
