# Mini-Projet : Sécurisation des images 2D par tatouage numérique basé sur QIM

Projet Python de tatouage numérique invisible et robuste basé sur :
- la transformation DCT 2D
- l'insertion d'un watermark binaire par QIM
- l'extraction et l'évaluation après attaques

## Fonctionnalités
- Chargement d'une image hôte en niveaux de gris
- Génération d'un watermark binaire pseudo-aléatoire
- Insertion du watermark dans des coefficients DCT de moyenne fréquence
- Sélection pseudo-aléatoire des positions via une clé secrète
- Simulation d'attaques :
  - bruit gaussien
  - compression JPEG
- Extraction du watermark
- Calcul des métriques :
  - PSNR
  - BER
- Sauvegarde d'images et d'un graphique de comparaison

## Structure

```bash
qim-watermark-project/
├── requirements.txt
├── README.md
├── main.py
├── .gitignore
└── src/
    └── qim_watermark/
        ├── __init__.py
        ├── io_utils.py
        ├── dct_utils.py
        ├── watermark.py
        ├── attacks.py
        └── metrics.py
```

## Installation

```bash
python -m venv .venv
```

### Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS
```bash
source .venv/bin/activate
```

Ensuite :

```bash
pip install -r requirements.txt
```

## Exemple d'exécution

```bash
python main.py --input host.png --watermark-size 1024 --delta 18 --seed 1234 --jpeg-quality 50 --noise-sigma 8
```

## Résultats générés
Le programme crée automatiquement un dossier `outputs/` avec :
- `watermarked.png`
- `attacked_noise.png`
- `attacked_jpeg.png`
- `comparison.png`
- `metrics.txt`

## Hébergement sur GitHub

```bash
git init
git add .
git commit -m "Initial commit - QIM DCT watermarking project"
git branch -M main
git remote add origin https://github.com/TON-USERNAME/qim-watermarking.git
git push -u origin main
```

## Idée du pipeline
1. Lire l'image hôte
2. Calculer la DCT 2D
3. Choisir des coefficients de moyenne fréquence
4. Insérer les bits par QIM
5. Reconstruire l'image tatouée via IDCT
6. Simuler des attaques
7. Extraire le watermark
8. Mesurer PSNR et BER

## Remarques
- Plus `delta` est grand, plus la robustesse augmente, mais la qualité visuelle peut baisser.
- Les coefficients DC et très basses fréquences sont évités.
- La même clé (`seed`) doit être utilisée pour l'insertion et l'extraction.
