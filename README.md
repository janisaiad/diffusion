# Dynamical Regimes of Diffusion Models

Ce projet explore les régimes dynamiques des modèles de diffusion, en se concentrant sur l'analyse théorique et expérimentale des différentes phases temporelles dans le processus de diffusion et débruitage.

![Diffusion Process](https://github.com/janisaiad/Dynamical-Regimes-of-Diffusion-Models/raw/main/beamer_reports/images/diffusion_process.png)

## 📋 Table des matières

- [Introduction](#introduction)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Résultats principaux](#résultats-principaux)
- [Commandes utiles](#commandes-utiles)
- [Pour contribuer](#pour-contribuer)
- [Licence](#licence)

## 🔍 Introduction

Les modèles de diffusion sont devenus des outils puissants en génération d'images et de signaux. Ce projet étudie en profondeur les régimes dynamiques qui apparaissent lors du processus de diffusion, avec une attention particulière sur :

- Les temps caractéristiques des transitions de phase 
- L'analyse du temps de collapse ($t_c$) et ses dépendances
- L'étude du temps de séparation ($t_s$) et son comportement
- La relation entre dimension, variance et nombre d'échantillons

Ces recherches ont des applications importantes pour la compréhension et l'optimisation des processus de diffusion dans différents domaines comme la génération d'images, le traitement du signal et l'apprentissage profond.

## 💻 Installation

### Prérequis

- Python 3.9+
- CUDA 12.x (pour l'accélération GPU)

### Installation des dépendances

```bash
# Cloner le dépôt
git clone https://github.com/janisaiad/Dynamical-Regimes-of-Diffusion-Models.git
cd Dynamical-Regimes-of-Diffusion-Models

# Installation avec uv (recommandé)
uv venv
source .venv/bin/activate
uv add -r pyproject.toml

# Pour installer les versions CUDA spécifiques
uv add cupy-cuda12x
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🗂️ Structure du projet

```
Dynamical-Regimes-of-Diffusion-Models/
├── data/                  # Données et datasets
├── models/                # Modèles spécifiques par expérience
│   ├── cifar/             # Expériences sur CIFAR
│   ├── MNIST/             # Expériences sur MNIST
│   ├── collapsetime/      # Analyse du temps de collapse
│   ├── hierarchical/      # Modèles hiérarchiques
│   ├── linear/            # Modèles linéaires
│   └── potentials/        # Analyse des potentiels
├── beamer_reports/        # Présentation Beamer des résultats
├── former/                # Code des versions précédentes
├── refs/                  # Références et articles scientifiques
├── results/               # Résultats d'expériences
├── tests/                 # Tests unitaires
├── launch.sh              # Script de lancement
├── pyproject.toml         # Configuration du projet et dépendances
├── README.md              # Ce fichier
└── report.tex             # Rapport LaTeX détaillé
```

## 🚀 Utilisation

### Scripts importants

- `launch.sh` : Script principal pour lancer les expériences
- `models/*/train.py` : Scripts d'entraînement pour différents datasets
- `models/*/analyze.py` : Scripts d'analyse des résultats


## 🛠️ Commandes utiles

```bash
# Obtenir toutes les branches distantes
git fetch --all --prune && git branch -r | grep -v '\->' | while read remote; do git branch --track "${remote#origin/}" "$remote" 2>/dev/null; done && git pull --all

# Installer les dépendances CUDA
uv add cupy-cuda12x
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Lancer les expériences MNIST
./launch.sh mnist

# Lancer les expériences CIFAR
./launch.sh cifar
```

## 🤝 Pour contribuer

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez votre branche de fonctionnalité (`git checkout -b feature/amazing-feature`)
3. Committez vos changements (`git commit -m 'Add amazing feature'`)
4. Poussez vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

---

Créé et maintenu par [Janis AIAD](https://github.com/janisaiad) dans le cadre du projet EAP2.
