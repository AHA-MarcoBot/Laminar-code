Code for our defense method Laminar.

Python packages:

tensorflow-gpu   2.8.0

python      3.8.13

keras         2.7.0

Full list of dependencies is listed in requirements.txt. Datasets used in this experiment can be downloaded from github pages of original papers.

Instructions for reproducing expirments (AWFdata/DFmodel for example):

1. Put dataset tor_200w_2500tr.npz in ./dataset/Closed World/, run get_burst.py;

2. Run train_substitute_model.py;

3. Run generate_universal.py, generated data is placed at ./advdata/train.dill & ./advdata/test.dill

4. For morphing ability ASR and MR evaluation, run evaluate_plain.py;

5. For AT defense under simple attack senario, switch seed in generate_universal.py and generate two batches of data, run evaluate_adversarial.py with data placed in corresponding location;

6. For AT defense under advanced attack senario, generate 5 batches of data and place them in corresponding locations (mentioned in evaluate_adversarial_advance.py) and run evaluate_adversarial_advance.py.
