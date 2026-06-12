# Journal des Performances et Suivi du Code : Micro-Beamforming

- **Date de référence :** 21 mai 2026
- **Configuration de test :** Serveur SRV07
- **Données de test :** 1 Frame, 960 Éléments, 30 SAPs au total

## 1. Phase d’Exploration : Journal des Performances Initiales

Ce premier tableau retrace les premiers essais du stage. Les temps ont été mesurés en lançant le script une seule fois, sans exécution à vide au démarrage (pas de warm-up). Ce tableau prend donc en compte le temps que met Python à charger les bibliothèques et à compiler le code la toute première fois.

| Version | Env. | Techno | Temps | Speedup baseline | Speedup MATLAB | Réduction | Analyse | Suite |
|---|---|---|---:|---:|---:|---:|---|---|
| `beamform_sorbet_micro.m` | MATLAB | Référence | 28.87 s | N/A | Référence | N/A | Code original de référence du laboratoire. | Point de départ. |
| `beamform_sorbet_micro.py`<br>*(Baseline)* | Python CPU | `scipy.interpolate.interp1d` | 19.45 s | 1.00x | 1.48x | 0.00 % | Traduction ligne par ligne du code MATLAB.<br>Corrélation : 0.999995. | Profilage avec `cProfile`. |
| Profilage baseline | Analyse CPU | `cProfile` + `matplotlib` | N/A | N/A | N/A | N/A | L’interpolation SciPy prend plus de 7 s à cause de la création répétée d’objets. | Remplacer par `numpy.interp`. |
| Optimisation 1 | Python CPU | `numpy.interp` | 15.90 s | 1.22x | 1.82x | 18.25 % | NumPy évite les objets lourds de SciPy. | Nouveau profilage. |
| Profilage opt. 1 | Analyse CPU | `cProfile` + CSV | N/A | N/A | N/A | N/A | La boucle `for` principale prend 12.67 s sur 30 720 répétitions. | Essai avec Threads. |
| Optimisation 2 | Python CPU MT | `ThreadPoolExecutor` | 18.92 s | 0.86x | 1.53x | -34.46 % | Régression à cause du GIL. | Essai Numba. |
| Profilage threads | Analyse CPU | `cProfile` threads + CSV | N/A | N/A | N/A | N/A | `wait` et `lock.acquire` cumulent plus de 7 s d’attente. | Passage à `@njit(parallel=True)`. |
| Optimisation 3 | Python CPU JIT | `numba` (`@njit(parallel=True)`) | 30.94 s | 0.63x | 0.93x | -59.07 % | Le premier lancement inclut la compilation JIT. | Diagnostic Numba. |
| Diag. Numba | Analyse CPU | `parallel_diagnostics(level=4)` | N/A | N/A | N/A | N/A | Échec de fusion de boucles (`fusion failed`), saturation mémoire. | Passage au GPU. |
| État de l’art | Analyse | Outils GPU | N/A | N/A | N/A | N/A | Étude de JAX, cuPyNumeric et CuPy. | Choix de CuPy. |
| Optimisation 5 | Python GPU | `cupy` float32 | 3.74 s | 5.20x | 7.72x | 80.77 % | Calcul déplacé sur GPU avec précision simple. | Profilage GPU. |
| Étape de profilage | Analyse CPU/GPU | `cProfile` + PNG | Analyse | N/A | N/A | N/A | `cupy.asarray` prend 2.159 s, interpolation GPU 0.408 s. | Essai des streams CUDA. |
| Optimisation finalisée | Python GPU async | `cupy.cuda.Stream` | 2.60 s | 7.48x | 11.10x | 90.99 % | Chevauchement transfert/calcul avec streams. | Nettoyage des boucles. |
| Optimisation VRAM fine | Python GPU async v2 | `cp.ascontiguousarray` | 2.42 s | 8.08x | 11.93x | 91.62 % | `arange` sort de la boucle, données contiguës. | Fin de l’étude mono-GPU. |
| Multi-GPU A | Windows | `cuPyNumeric` | Bloqué | N/A | N/A | N/A | Installation impossible sur Windows natif. | Test sous WSL2. |
| Multi-GPU B | Linux WSL2 | `cuPyNumeric` | Abandon | N/A | N/A | N/A | `CUDA_ERROR_OUT_OF_MEMORY` et besoin de Python 3.10+. | Piste abandonnée. |
| Multi-GPU C | Windows | `JAX` | Échec | N/A | N/A | N/A | `TypeError: object does not support item assignment`. | Piste écartée. |
| Perspectives | Python / C++ | Kernels CUDA custom | À déterminer | N/A | N/A | N/A | Fusionner indices, décalages et interpolation dans un kernel CUDA unique. | Ouverture rapport. |

## 2. Validation des méthodes de mesure temporelle

Pour assurer la validité des métriques, une étude comparative des outils de chronométrage intégrés à Python a été menée afin de sélectionner l’indicateur le plus stable face aux variations de l’ordinateur d’exploitation.

| Outil | Documentation | Validé | Raison |
|---|---|---|---|
| `time.time()` | [`time`](https://docs.python.org/3/library/time.html#time.time) | Oui | Donne l’heure système, mais peut être affecté par une resynchronisation de l’horloge. |
| `timeit.timeit()` | [`timeit`](https://docs.python.org/3/library/timeit.html) | Non | Adapté aux petites portions de code, pas à un algorithme complet. |
| `time.process_time()` | [`time`](https://docs.python.org/3/library/time.html#time.process_time) | Non | Ne mesure pas le temps GPU. |
| `time.perf_counter()` | [`time`](https://docs.python.org/3/library/time.html#time.perf_counter) | Oui | Compteur précis et monotone, retenu comme choix final. |

### Calculs utilisés pour les indicateurs de performance

- **Speedup global (vs MATLAB)** :  
  \( \text{Speedup}_{\text{global}} = \frac{T_{\text{MATLAB}}}{T_{\text{Version}}} \)

- **Gain vs Baseline Python** :  
  \( \text{Gain}_{\text{vs\_baseline}} = \frac{T_{\text{Baseline Python}}}{T_{\text{Version Optimisée}}} \)

- **Réduction du temps de calcul** :  
  \( \text{Réduction} = \frac{T_{\text{Initial}} - T_{\text{Optimisé}}}{T_{\text{Initial}}} \times 100 \)

## 3. Ressources documentaires et veille technique

Cette section rassemble les documents, liens officiels et présentations d’ingénieurs utilisés pendant le stage pour comprendre les blocages techniques rencontrés.

### Fonctionnement des threads en Python
- **GIL (Global Interpreter Lock)** : mécanisme interne de Python qui empêche plusieurs threads d’exécuter du bytecode Python en parallèle.  
  Référence : [Documentation officielle du GIL Python](https://docs.python.org/3/glossary.html#term-GIL)

### Mesure et optimisation
- **Boucle de benchmark du CNES** : le protocole utilisé s’inspire de l’atelier HPC du CNES.  
  Référence : [Support CNES](https://sourcesup.renater.fr/wiki/atelieromp/_media/20221201_atelier_omp_optimisation_code_python.pdf)
- **Documentation `timeit`** : explique pourquoi le minimum est souvent la métrique la plus fiable pour l’évaluation de performance.  
  Référence : [Documentation Python `timeit`](https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat)
- **Guide de propreté du code** : utilisé pour le nettoyage et la réorganisation du script.  
  Référence : [Guide S. Robert](https://blog.stephane-robert.info/docs/developper/programmation/python/linting/)

### Outils testés pour le Multi-GPU
- **cuPyNumeric** : outil NVIDIA étudié pour la distribution de calculs sur plusieurs GPU.  
  Référence : [Dépôt officiel cuPyNumeric](https://github.com/nv-legate/cupynumeric)

## 4. Réorganisation du code et bilan final des performances

Après les essais exploratoires, les scripts ont été regroupés dans une structure plus simple et plus propre pour l’équipe.

Le code a été séparé en deux fichiers :
1. `src/beamformer_engines.py` : fonctions de calcul pur.
2. `src/beamformer.py` : classe `Beamformer` avec appel `.das(rf, method='...')`.

### Protocole de mesure retenu

Le script final utilise un protocole inspiré du CNES :
- Un warm-up non chronométré.
- Une boucle de 5 répétitions.
- Conservation du temps minimum.

### Tableau comparatif après réorganisation du code

Grâce à la centralisation de l’architecture, à la suppression des allocations répétitives dans les boucles et au nettoyage des variables intermédiaires, les temps de traitement ont été réduits sur tous les moteurs de calcul.

| Rang | Moteur | Matériel | Temps | Speedup MATLAB | Gain vs Baseline | Réduction vs MATLAB |
|---|---|---|---:|---:|---:|---:|
| Réf | MATLAB | CPU standard | 28.87 s | 1.00x | N/A | Référence |
| #1 | `gpu_block` | GPU CUDA standard | 1.67 s | 17.31x | 6.33x | -94.22 % |
| #2 | `gpu_streams` | GPU CUDA + streams | 1.71 s | 16.90x | 6.18x | -94.08 % |
| #3 | `numba` | CPU JIT parallèle | 8.15 s | 3.54x | 1.29x | -71.75 % |
| #4 | `numpy` | CPU vectorisé | 8.70 s | 3.32x | 1.21x | -69.85 % |
| #5 | `threads` | CPU multi-threading | 9.25 s | 3.12x | 1.14x | -67.95 % |
| #6 | `scipy` | CPU mono-thread | 10.55 s | 2.74x | 1.00x | -63.45 % |

### Analyse et interprétation des résultats

1. **Impact du nettoyage de la structure** : la version `scipy` passe de 19.45 s à 10.55 s grâce à la réorganisation du code.
2. **Correction du temps Numba** : le warm-up retire le coût de la compilation initiale.
3. **Comparaison GPU** : `gpu_block` reste légèrement devant `gpu_streams` sur une seule frame.
4. **Validation des données** : toutes les méthodes produisent des sorties conformes à MATLAB avec une corrélation finale de 0.99995.

L’implémentation finale GPU répond aux objectifs du stage avec un gain global de 17.31x par rapport à la version d’origine.
