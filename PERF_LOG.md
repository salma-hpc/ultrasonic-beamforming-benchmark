# Journal des Performances et Suivi du Code : Micro-Beamforming

- **Date de référence :** 21 mai 2026
- **Configuration de test :** Serveur SRV07
- **Données de test :** 1 Frame, 960 Éléments, 30 SAPs au total

## 1. Phase d’Exploration : Journal des Performances Initiales

Ce premier tableau retrace les premiers essais du stage. Les temps ont été mesurés en lançant le script une seule fois, sans exécution à vide au démarrage (pas de warm-up). Ce tableau prend donc en compte le temps que met Python à charger les bibliothèques et à compiler le code la toute première fois.

| Version du code | Environnement | Technologie | Temps de calcul | Speedup vs Baseline Python | Speedup global vs MATLAB | Réduction vs Baseline | Analyse technique | Étape suivante |
|---|---|---|---:|---:|---:|---:|---|---|
| `beamform_sorbet_micro.m` | MATLAB | Code original de référence | 28.87 s | N/A | Référence absolue | N/A | Code de base fourni par le laboratoire pour vérifier que les futures versions Python donnent les mêmes résultats. | Point de départ de l’étude. |
| `beamform_sorbet_micro.py` *(Baseline)* | Python (CPU) | `scipy.interpolate.interp1d` | 19.45 s | 1.00x | 1.48x | 0.00 % | Traduction ligne par ligne du code MATLAB en Python. Les résultats sont validés par un calcul de corrélation (0.999995). | Lancement d’un premier profilage avec cProfile. |
| Profilage de la Baseline | Analyse CPU | `cProfile` + `matplotlib` | N/A (Analyse) | N/A | N/A | N/A | La fonction d’interpolation de SciPy (`__init__`, `__call__`) prend plus de 7 secondes à elle seule à cause de la création répétée d’objets. | Remplacement par `numpy.interp`. |
| Optimisation 1 | Python (CPU) | Remplacement par `numpy.interp` | 15.90 s | 1.22x | 1.82x | 18.25 % | Le calcul direct avec NumPy évite la création d’objets lourds. | Nouveau profilage. |
| Profilage de l’Optimisation 1 | Analyse CPU | Profilage incrémental (`cProfile` + CSV) | N/A (Analyse) | N/A | N/A | N/A | Le problème vient maintenant de la boucle `for` principale en Python, répétée 30 720 fois, qui prend 12.67 s. | Essai de parallélisation avec Threads. |
| Optimisation 2 | Python (CPU Multi-threads) | `ThreadPoolExecutor` | 18.92 s | 0.86x *(Régression)* | 1.53x | -34.46 % | Les threads se bloquent à cause du GIL. | Essai de Numba. |
| Profilage du Multi-threading | Analyse CPU Parallèle | `cProfile` dédié Threads + CSV | N/A (Analyse) | N/A | N/A | N/A | Les fonctions de blocage (`wait`, `lock.acquire`) cumulent plus de 7 secondes d’attente. | Passage à `@njit(parallel=True)`. |
| Optimisation 3 | Python (CPU JIT Parallèle) | `numba (@njit(parallel=True))` | 30.94 s | 0.63x *(Régression)* | 0.93x | -59.07 % | Le premier lancement inclut le temps de compilation Numba. | Utilisation de `parallel_diagnostics`. |
| Profilage & Diagnostic Numba | Analyse CPU | `parallel_diagnostics(level=4)` | N/A (Analyse) | N/A | N/A | N/A | La fusion automatique des boucles a échoué (`fusion failed`). | Passage au GPU. |
| Étude de l’état de l’art | Analyse théorique | Outils GPU (CNRS) | N/A (Analyse) | N/A | N/A | N/A | Étude de JAX, cuPyNumeric et CuPy. | Choix de CuPy. |
| Optimisation 5 | Python (GPU massif) | `cupy` (float32) | 3.74 s | 5.20x | 7.72x | 80.77 % | Passage des calculs sur la carte graphique. | Profilage GPU. |
| Étape de Profilage | Analyse CPU/GPU | `cProfile` + PNG | Analyse | N/A | N/A | N/A | L’envoi des données vers le GPU (`cupy.asarray`) prend le plus de temps. | Essai des streams CUDA. |
| Optimisation Finalisée | Python (GPU asynchrone) | `cupy.cuda.Stream` | 2.60 s | 7.48x | 11.10x | 90.99 % | Utilisation des Streams pour chevaucher transfert et calcul. | Nettoyage des boucles. |
| Optimisation VRAM Fine | Python (GPU asynchrone v2) | `cp.ascontiguousarray` + nettoyage boucle | 2.42 s | 8.08x | 11.93x | 91.62 % | `arange` sort de la boucle, données contiguës en mémoire. | Fin de l’étude sur une seule carte. |
| Évaluation Multi-GPU (Piste 2 - Essai A) | Windows (local) | `cuPyNumeric` | N/A (Bloqué) | N/A | N/A | N/A | Incompatible avec Windows natif. | Tentative sous WSL2. |
| Évaluation Multi-GPU (Piste 2 - Essai B) | Linux (WSL2) | `cuPyNumeric` | Abandon | N/A | N/A | N/A | Erreurs mémoire et contrainte Python 3.10+. | Piste abandonnée. |
| Évaluation Multi-GPU (Piste 3) | Windows (local) | `JAX` | Échec (Syntaxe) | N/A | N/A | N/A | `TypeError: object does not support item assignment`. | Piste écartée. |
| Perspectives d’avenir | Python / C++ | Kernels CUDA personnalisés | À déterminer | N/A | N/A | N/A | Regrouper indices, décalages et interpolation dans un kernel CUDA. | Ouverture pour la suite. |

---

## 2. Validation des méthodes de mesure temporelle

Pour assurer la validité des métriques, une étude comparative des outils de chronométrage intégrés à Python a été menée afin de sélectionner l’indicateur le plus stable face aux variations de l’ordinateur d’exploitation.

| Outil de mesure | Bibliothèque / Documentation | Validé | Raison du choix |
|---|---|---|---|
| `time.time()` | [`time`](https://docs.python.org/3/library/time.html#time.time) | Oui | Donne l’heure de l’ordinateur. |
| `timeit.timeit()` | [`timeit`](https://docs.python.org/3/library/timeit.html) | Non | Fait pour des micro-boucles. |
| `time.process_time()` | [`time`](https://docs.python.org/3/library/time.html#time.process_time) | Non | Ne mesure pas le temps GPU. |
| `time.perf_counter()` | [`time`](https://docs.python.org/3/library/time.html#time.perf_counter) | Oui | Choix final. |

### Calculs utilisés pour les indicateurs de performance

- **Speedup global (par rapport à MATLAB) :**
  \(\text{Speedup}_{\text{global}} = \frac{T_{\text{MATLAB}}}{T_{\text{Version}}}\)

- **Gain vs Baseline Python :**
  \(\text{Gain}_{\text{vs\_baseline}} = \frac{T_{\text{Baseline Python}}}{T_{\text{Version Optimisée}}}\)

- **Taux de réduction du temps de calcul :**
  \(\text{Réduction} = \frac{T_{\text{Initial}} - T_{\text{Optimisé}}}{T_{\text{Initial}}} \times 100\)

---

## 3. Ressources documentaires et veille technique

Cette section rassemble les documents, liens officiels et présentations d’ingénieurs utilisés pendant le stage pour comprendre les blocages techniques rencontrés.

### Fonctionnement des threads en Python
- **Le mécanisme du GIL (Global Interpreter Lock) :** règle interne de Python qui interdit à plusieurs threads d’exécuter du code en même temps.  
  Référence : [Documentation officielle du GIL Python](https://docs.python.org/3/glossary.html#term-GIL)

### Méthodes pour mesurer et optimiser le code
- **Exemple de boucle de benchmark du CNES :** le protocole mis en place pour nettoyer les mesures s’inspire du code d’exemple de leur atelier HPC.  
  Référence : [Support de cours du Centre de Calcul du CNES](https://sourcesup.renater.fr/wiki/atelieromp/_media/20221201_atelier_omp_optimisation_code_python.pdf)
- **Documentation officielle du module `timeit` :** explique pourquoi la valeur minimale reste la mesure la plus fiable.  
  Référence : [Documentation Python - `timeit`](https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat)
- **Guide de propreté du code :** conseils suivis pour organiser le script.  
  Référence : [Analyse et Optimisation de code Python - S. Robert](https://blog.stephane-robert.info/docs/developper/programmation/python/linting/)

### Documentations des outils testés pour le Multi-GPU
- **cuPyNumeric de NVIDIA :** outil testé sous Windows puis abandonné suite aux incompatibilités.  
  Référence : [Dépôt officiel cuPyNumeric](https://github.com/nv-legate/cupynumeric)

---

## 4. Réorganisation du code et bilan final

Après avoir testé les différentes technologies, les scripts de recherche ont été regroupés dans une structure de fichiers plus propre et plus simple à utiliser pour l’équipe.

Le code a été séparé en deux fichiers :
1. `src/beamformer_engines.py` : fonctions de calcul pur.
2. `src/beamformer.py` : classe `Beamformer` avec la méthode `.das(rf, method='...')`.

### Protocole de mesure retenu
Le script final utilise un protocole inspiré de la page 38 du document d’optimisation du CNES :
- Un warm-up non chronométré.
- Une boucle de 5 répétitions.
- Conservation du temps minimum.

### Tableau comparatif après réorganisation

Grâce à la centralisation de l’architecture, à la suppression des allocations répétitives et au nettoyage des variables intermédiaires, les temps de traitement ont été réduits sur tous les moteurs de calcul.

| Rang | Moteur | Environnement | Temps d’exécution | Speedup vs MATLAB | Gain vs Baseline Python | Réduction vs MATLAB |
|---|---|---|---:|---:|---:|---:|
| Réf | MATLAB (code original) | CPU standard | 28.87 s | 1.00x | N/A | Référence |
| #1 | `gpu_block` | GPU CUDA standard | 1.67 s | 17.31x | 6.33x | -94.22 % |
| #2 | `gpu_streams` | GPU CUDA + flux asynchrones | 1.71 s | 16.90x | 6.18x | -94.08 % |
| #3 | `numba` | CPU JIT parallèle | 8.15 s | 3.54x | 1.29x | -71.75 % |
| #4 | `numpy` | CPU vectorisé | 8.70 s | 3.32x | 1.21x | -69.85 % |
| #5 | `threads` | CPU multi-threading | 9.25 s | 3.12x | 1.14x | -67.95 % |
| #6 | `scipy` | CPU mono-thread | 10.55 s | 2.74x | 1.00x | -63.45 % |

### Analyse et interprétation des résultats

1. **Impact du nettoyage de la structure :** la version de base `scipy` passe de 19.45 s à 10.55 s.
2. **Correction du temps Numba :** le warm-up retire le temps de première compilation.
3. **Comparaison des moteurs GPU :** le mode bloc reste très proche du mode streams.
4. **Validation des données :** toutes les méthodes ont été contrôlées avec une corrélation finale de 0.99995.

L’implémentation finale sur l’architecture GPU permet de répondre aux objectifs fixés par Vermon, avec un gain global de 17.31x par rapport à la version d’origine.
